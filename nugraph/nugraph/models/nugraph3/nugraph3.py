"""NuGraph3 model architecture"""
import argparse
import warnings

import torch.cuda
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from pytorch_lightning import LightningModule

from .types import Data
from .transform import Transform
from .encoder import Encoder
from .core import NuGraphCore
from .decoders import (SemanticDecoder, FilterDecoder, EventDecoder, VertexDecoder, InstanceDecoder,
                       SpacepointDecoder)

from ...data import H5DataModule

class NuGraph3(LightningModule):
    """
    NuGraph3 model architecture.

    Args:
        in_features: Number of input node features
        hit_features: Number of hit node features
        nexus_features: Number of nexus node features
        interaction_features: Number of interaction node features
        instance_features: Number of instance features
        planes: Tuple of detector plane names
        semantic_classes: Tuple of semantic classes
        event_classes: Tuple of event classes
        num_iters: Number of message-passing iterations
        event_head: Whether to enable event decoder
        semantic_head: Whether to enable semantic decoder
        filter_head: Whether to enable filter decoder
        vertex_head: Whether to enable vertex decoder
        instance_head: Whether to enable instance decoder
        spacepoint_head: Whether to enable spacepoint decoder
        use_checkpointing: Whether to use checkpointing
        lr: Learning rate
        no_one_cycle_sched: Whether to disable the OneCycleLR scheduler
    """
    def __init__(self,
                 in_features: int = 4,
                 hit_features: int = 128,
                 nexus_features: int = 32,
                 interaction_features: int = 32,
                 instance_features: int = 8,
                 planes: tuple[str] = ("u","v","y"),
                 semantic_classes: tuple[str] = ('MIP','HIP','shower','michel','diffuse'),
                 event_classes: tuple[str] = ('numu','nue','nc'),
                 num_iters: int = 5,
                 event_head: bool = False,
                 semantic_head: bool = True,
                 filter_head: bool = True,
                 vertex_head: bool = False,
                 instance_head: bool = False,
                 spacepoint_head: bool = False,
                 particle_loss: bool = False,
                 use_checkpointing: bool = False,
                 lr: float = 0.001,
                 no_one_cycle_sched: bool = False,
                 materializer: str = "dbscan",
                 materialize_train: bool = False,
                 materialize_every_n: int = 1,
                 mat_t_beta: float = 0.1,
                 mat_t_d: float = 0.3,
                 mat_min_cluster_size: int = 1,
                 dbscan_eps: float = 0.3,
                 dbscan_min_samples: int = 15,
                 materialize_timing: bool = False,
                 mat_max_pairs: int = 4_000_000,
                 mat_max_seeds: int = 20_000,
                 mat_max_edges: int = 8_000_000,
                 mat_cc_max_iters: int = 200,
                 mat_max_clusters: int = 20_000,
                 mat_on_unsafe: str = "flag",
                 mat_fallback: str = "dbscan",
                 materialize_metric: bool = True):
        super().__init__()

        warnings.filterwarnings("ignore", ".*NaN values found in confusion matrix.*")

        self.save_hyperparameters()

        self.nexus_features = nexus_features
        self.interaction_features = interaction_features

        self.semantic_classes = semantic_classes
        self.event_classes = event_classes
        self.num_iters = num_iters
        self.lr = lr
        self.no_one_cycle_sched = no_one_cycle_sched

        self.encoder = Encoder(in_features, hit_features,
                               nexus_features, interaction_features, instance_features)

        self.core_net = NuGraphCore(hit_features,
                                    nexus_features,
                                    interaction_features,
                                    instance_features,
                                    use_checkpointing)

        self.decoders = []

        if event_head:
            self.event_decoder = EventDecoder(interaction_features, event_classes)
            self.decoders.append(self.event_decoder)

        if semantic_head:
            self.semantic_decoder = SemanticDecoder(hit_features, semantic_classes)
            self.decoders.append(self.semantic_decoder)

        if filter_head:
            self.filter_decoder = FilterDecoder(hit_features,)
            self.decoders.append(self.filter_decoder)

        if vertex_head:
            self.vertex_decoder = VertexDecoder(interaction_features)
            self.decoders.append(self.vertex_decoder)

        if instance_head:
            self.instance_decoder = InstanceDecoder(
                hit_features, instance_features, particle_loss,
                materializer=materializer,
                materialize_train=materialize_train,
                materialize_every_n=materialize_every_n,
                t_beta=mat_t_beta,
                t_d=mat_t_d,
                min_cluster_size=mat_min_cluster_size,
                dbscan_eps=dbscan_eps,
                dbscan_min_samples=dbscan_min_samples,
                materialize_timing=materialize_timing,
                mat_max_pairs=mat_max_pairs,
                mat_max_seeds=mat_max_seeds,
                mat_max_edges=mat_max_edges,
                mat_cc_max_iters=mat_cc_max_iters,
                mat_max_clusters=mat_max_clusters,
                mat_on_unsafe=mat_on_unsafe,
                mat_fallback=mat_fallback,
                materialize_metric=materialize_metric)
            self.decoders.append(self.instance_decoder)

        if spacepoint_head:
            self.spacepoint_decoder = SpacepointDecoder(hit_features, len(planes))
            self.decoders.append(self.spacepoint_decoder)

        if not self.decoders:
            raise RuntimeError('At least one decoder head must be enabled!')

    def forward(self, data: Data, stage: str = None): # pylint: disable=arguments-differ
        """
        NuGraph3 forward function

        This function runs the forward pass of the NuGraph3 architecture,
        and then loops over each decoder to compute the loss and calculate
        and log any performance metrics.

        Args:
            data: Graph data object
            stage: String tag defining the step type
        """
        self.encoder(data)
        for _ in range(self.num_iters):
            self.core_net(data)
        total_loss = 0.
        total_metrics = {}
        for decoder in self.decoders:
            loss, metrics = decoder(data, stage)
            total_loss += loss
            total_metrics.update(metrics)

        return total_loss, total_metrics

    def training_step(self,
                      batch: Data,
                      batch_idx: int) -> float:
        loss, metrics = self(batch, 'train')
        self.log('loss/train', loss, batch_size=batch.num_graphs, prog_bar=True, sync_dist=True)
        self.log_dict(metrics, batch_size=batch.num_graphs, sync_dist=True)
        return loss

    def on_train_epoch_end(self) -> None:
        # stop updating running average for feature norm
        self.encoder.input_norm.update = False

    def validation_step(self,
                        batch,
                        batch_idx: int) -> None:
        loss, metrics = self(batch, 'val')
        self.log('loss/val', loss, batch_size=batch.num_graphs, sync_dist=True)
        self.log_dict(metrics, batch_size=batch.num_graphs, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        epoch = self.trainer.current_epoch + 1
        for decoder in self.decoders:
            decoder.on_epoch_end(self.logger, 'val', epoch)

    def test_step(self,
                  batch,
                  batch_idx: int = 0) -> None:
        loss, metrics = self(batch, 'test')
        self.log('loss/test', loss, batch_size=batch.num_graphs, sync_dist=True)
        self.log_dict(metrics, batch_size=batch.num_graphs, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        epoch = self.trainer.current_epoch + 1
        for decoder in self.decoders:
            decoder.on_epoch_end(self.logger, 'test', epoch)

    def predict_step(self,
                     batch: Data,
                     batch_idx: int = 0) -> Data:
        self(batch)
        return batch

    def configure_optimizers(self) -> tuple:
        optimizer = AdamW(self.parameters(),
                          lr=self.lr)
        if self.no_one_cycle_sched:
            return optimizer
        else:
            onecycle = OneCycleLR(
                       optimizer,
                       max_lr=self.lr,
                       total_steps=self.trainer.estimated_stepping_batches)
            return [optimizer], {'scheduler': onecycle, 'interval': 'step'}

    @staticmethod
    def transform(planes: tuple[str]) -> Transform:
        """
        Return data transform for NuGraph3 model

        Args:
            planes: tuple of detector plane names
        """
        return Transform(planes)

    @staticmethod
    def add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Add argparse argument group for NuGraph3 model

        Args:
            parser: Argument parser to append argument group to
        """
        model = parser.add_argument_group('model', 'NuGraph3 model configuration')
        model.add_argument('--num-iters', type=int, default=5,
                           help='Number of message-passing iterations')
        model.add_argument('--in-feats', type=int, default=5,
                           help='Number of input node features')
        model.add_argument('--hit-feats', type=int, default=128,
                           help='Hidden dimensionality of hit convolutions')
        model.add_argument('--nexus-feats', type=int, default=32,
                           help='Hidden dimensionality of nexus convolutions')
        model.add_argument('--interaction-feats', type=int, default=32,
                           help='Hidden dimensionality of interaction layer')
        model.add_argument('--instance-feats', type=int, default=8,
                           help='Hidden dimensionality of object condensation')
        model.add_argument('--event', action='store_true',
                           help='Enable event classification head')
        model.add_argument('--semantic', action='store_true',
                           help='Enable semantic segmentation head')
        model.add_argument('--filter', action='store_true',
                           help='Enable background filter head')
        model.add_argument('--instance', action='store_true',
                           help='Enable instance segmentation head')
        model.add_argument('--vertex', action='store_true',
                           help='Enable vertex regression head')
        model.add_argument("--spacepoint", action="store_true",
                           help="Enable spacepoint prediction head")
        model.add_argument("--particle-loss", action="store_true",
                           help="Enable object condensation particle loss term")
        model.add_argument('--no-checkpointing', action='store_false',
                           dest="use_checkpointing",
                           help='Disable checkpointing during training')
        model.add_argument('--epochs', type=int, default=80,
                           help='Maximum number of epochs to train for')
        model.add_argument('--learning-rate', type=float, default=0.001,
                           help='Max learning rate during training')
        model.add_argument('--no-lr-scheduler', action='store_true',
                           dest="no_one_cycle_sched",
                           help='Disable OneCycleLR scheduler')

        # instance materialization backend selection + parameters
        mat = parser.add_argument_group(
            'materialization', 'Instance materialization backend configuration')
        mat.add_argument('--materializer', type=str, default='dbscan',
                         choices=('dbscan', 'seed', 'graph', 'greedy'),
                         help='Instance materialization backend '
                              '(dbscan=baseline, seed=A, graph=B, greedy=C)')
        mat.add_argument('--materialize-train', action='store_true',
                         help='Run materialization during training as well '
                              '(default: validation/test/inference only)')
        mat.add_argument('--materialize-every-n', type=int, default=1,
                         help='When training, materialize every N steps')
        mat.add_argument('--mat-t-beta', type=float, default=0.1,
                         help='Condensation-score seed threshold (seed/greedy)')
        mat.add_argument('--mat-t-d', type=float, default=0.3,
                         help='Latent radius for GPU backends (like DBSCAN eps)')
        mat.add_argument('--mat-min-cluster-size', type=int, default=1,
                         help='Drop clusters smaller than this (1 keeps all)')
        mat.add_argument('--dbscan-eps', type=float, default=0.3,
                         help='DBSCAN eps (dbscan backend)')
        mat.add_argument('--dbscan-min-samples', type=int, default=15,
                         help='DBSCAN min_samples (dbscan backend)')
        mat.add_argument('--materialize-timing', action='store_true',
                         help='Enable per-operation materialization timing')
        # memory-safety / high-occupancy limits (see materialize.py)
        mat.add_argument('--mat-max-pairs', type=int, default=4_000_000,
                         help='Max elements in any pairwise temp (memory budget)')
        mat.add_argument('--mat-max-seeds', type=int, default=20_000,
                         help='Cap on candidate seeds (seed backend); explicit '
                              'top-beta truncation, flagged if reached')
        mat.add_argument('--mat-max-edges', type=int, default=8_000_000,
                         help='Cap on radius-graph edges (graph backend)')
        mat.add_argument('--mat-cc-max-iters', type=int, default=200,
                         help='Connected-components iteration cap (graph backend)')
        mat.add_argument('--mat-max-clusters', type=int, default=20_000,
                         help='Cluster cap for the greedy backend')
        mat.add_argument('--mat-on-unsafe', type=str, default='flag',
                         choices=('flag', 'raise', 'skip', 'fallback'),
                         help='On an invalid/safety-limited result: flag+log '
                              '(default), raise, skip the batch, or re-run with '
                              '--mat-fallback backend')
        mat.add_argument('--mat-fallback', type=str, default='dbscan',
                         choices=('dbscan', 'seed', 'graph', 'greedy'),
                         help='Fallback backend when --mat-on-unsafe fallback')
        mat.add_argument('--no-materialize-metric', dest='materialize_metric',
                         action='store_false',
                         help='Throughput mode: skip the ARI quality metric '
                              'during materialization')
        return parser

    @classmethod
    def from_args(cls, args: argparse.Namespace, nudata: H5DataModule) -> 'NuGraph3':
        """
        Construct model from arguments

        Args:
            args: Namespace containing parsed arguments
            nudata: Data module
        """
        return cls(
            in_features=args.in_feats,
            hit_features=args.hit_feats,
            nexus_features=args.nexus_feats,
            interaction_features=args.interaction_feats,
            instance_features=args.instance_feats,
            planes=nudata.planes,
            semantic_classes=nudata.semantic_classes,
            event_classes=nudata.event_classes,
            num_iters=args.num_iters,
            event_head=args.event,
            semantic_head=args.semantic,
            filter_head=args.filter,
            vertex_head=args.vertex,
            instance_head=args.instance,
            spacepoint_head=args.spacepoint,
            particle_loss=args.particle_loss,
            use_checkpointing=args.use_checkpointing,
            lr=args.learning_rate,
            no_one_cycle_sched=args.no_one_cycle_sched,
            materializer=args.materializer,
            materialize_train=args.materialize_train,
            materialize_every_n=args.materialize_every_n,
            mat_t_beta=args.mat_t_beta,
            mat_t_d=args.mat_t_d,
            mat_min_cluster_size=args.mat_min_cluster_size,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
            materialize_timing=args.materialize_timing,
            mat_max_pairs=args.mat_max_pairs,
            mat_max_seeds=args.mat_max_seeds,
            mat_max_edges=args.mat_max_edges,
            mat_cc_max_iters=args.mat_cc_max_iters,
            mat_max_clusters=args.mat_max_clusters,
            mat_on_unsafe=args.mat_on_unsafe,
            mat_fallback=args.mat_fallback,
            materialize_metric=args.materialize_metric)
