from deepchem.models.losses import Loss, _make_pytorch_shapes_consistent


class AntibioticLoss(Loss):
    """Loss for virtual screening of antibiotics that uses an asymmetric sigmoid function,
    considering active if inhibition > 0.8 and inactive if inhibition < 0.8

    (Not squashed version)
    L(y,z) = - c_fp (1 - s(y)) log (1 + exp(z)) - c_fn s(y) log (1 - exp(-z))

    where
    - y is a real activity value (ranging mostly from 0 to 1),
    - z is the output of a neural network,
    - c_fp is a weighting coefficient for false positives, and
    - c_fn is the coefficient for false negatives.

    Note that the previous is equivalent to the more intuitive

    (Squashed version)
    L(y,ŷ) = - c_fp (1 - s(y)) log (1 - ŷ) - c_fn s(y) log ŷ

    where ŷ = 1 / (1 + exp(-z)). This expression is very similar to the
    cross-entropy loss. Therefore the antibiotic loss is a version of the
    cross-entropy loss where y is real instead of binary.
    """

    def __init__(self,c_fp=0.5,c_fn=0.5,squashed_input=False):
        super().__init__()
        self.c_fp = c_fp
        self.c_fn = c_fn
        self.squashed_input = squashed_input

    def _compute_tf_loss(self, output, labels):
        raise NotImplementedError

    def _create_pytorch_loss(self):
        import torch

        def asymmetric_sigmoid(x):
            return torch.sigmoid(4*(torch.exp(3.28*(x - 0.8)) - 1))

        def asymmetric_loss(output, labels):
            output, labels = _make_pytorch_shapes_consistent(output, labels)
            y = labels
            if self.squashed_input:
                hat_y = output
                false_positives =  - (1 - asymmetric_sigmoid(y)) * torch.log(1 - hat_y)
                false_negatives = - asymmetric_sigmoid(y) * torch.log(hat_y)
            else:
                z = output
                false_positives =  (1 - asymmetric_sigmoid(y)) * torch.log(1 + torch.exp(z))
                false_negatives = asymmetric_sigmoid(y) * torch.log(1 + torch.exp(-z))

            loss = self.c_fp * false_positives + self.c_fn * false_negatives
            return loss

        return asymmetric_loss

class REDLoss(Loss):
    """Loss for the red dataset (rdkit, excape, dockstring). Combines
    L2 loss for rdkit and dockstring with cross entropy for excape.
    """

    def __init__(self,r_slice,e_slice,d_slice):
        super().__init__()
        self.r_slice = r_slice
        self.e_slice = e_slice
        self.d_slice = d_slice

    def _create_pytorch_loss(self):
        import torch
        ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        l2_loss = torch.nn.MSELoss(reduction='none')

        def loss(output, labels):
            r_output = output[:, self.r_slice, 1]
            r_labels = labels[:, self.r_slice]
            e_output = output[:, self.e_slice]
            e_labels = labels[:, self.e_slice]
            d_output = output[:, self.d_slice, 1]
            d_labels = labels[:, self.d_slice]
            # Convert (batch_size, tasks, classes) to (batch_size, classes, tasks)
            # CrossEntropyLoss only supports (batch_size, classes, tasks)
            # This is for API consistency
            if len(e_output.shape) == 3:
                e_output = e_output.permute(0, 2, 1)
            if len(e_labels.shape) == len(e_output.shape):
                e_labels = e_labels.squeeze(-1)

            r_loss = l2_loss(r_output, r_labels)
            e_loss = ce_loss(e_output, e_labels.long()).float()
            d_loss = l2_loss(d_output, d_labels)
            return torch.cat([r_loss,e_loss,d_loss],axis=1)

        return loss
