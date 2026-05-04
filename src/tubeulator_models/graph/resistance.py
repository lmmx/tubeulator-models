import torch
from torch_geometric.utils import get_laplacian, to_dense_adj


def effective_resistance(edge_index, edge_weight=None):
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1))

    edge_weight = torch.clamp(edge_weight, min=1e-8)

    lap_idx, lap_w = get_laplacian(
        edge_index,
        edge_weight,
        normalization=None,
    )

    L = to_dense_adj(lap_idx, edge_attr=lap_w)[0]

    L = 0.5 * (L + L.T)

    L_pinv = torch.linalg.pinv(L)

    row, col = edge_index

    R = L_pinv[row, row] + L_pinv[col, col] - 2.0 * L_pinv[row, col]

    return torch.clamp(R, min=1e-8)


def node_curvature(edge_index, edge_weight, R):
    """p_i = 1 - 0.5 * sum_{j~i} c_ij * omega_ij"""
    N = int(edge_index.max()) + 1
    row = edge_index[0]

    # relative resistance = c_ij * omega_ij
    rel_R = edge_weight * R

    acc = torch.zeros(N, dtype=R.dtype, device=R.device)
    acc.index_add_(0, row, rel_R)

    return 1.0 - 0.5 * acc


def edge_curvature(edge_index, node_curv, R):
    """kappa_ij = 2(p_i + p_j) / omega_ij"""
    row, col = edge_index
    return 2.0 * (node_curv[row] + node_curv[col]) / R
