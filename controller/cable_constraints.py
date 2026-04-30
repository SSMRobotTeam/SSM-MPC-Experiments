import copy

import numpy as np


def _cell(node_list, section_idx, cable_idx, node_idx):
    return np.asarray(node_list[section_idx][cable_idx, node_idx]).reshape(-1)


def _set_cell_xyz(node_list, section_idx, cable_idx, node_idx, value):
    arr = np.asarray(node_list[section_idx][cable_idx, node_idx]).copy().reshape(-1)
    arr[:3] = value
    node_list[section_idx][cable_idx, node_idx] = arr


def cable_length_double_part(x, node_list):
    x = np.asarray(x, dtype=float).reshape(-1)
    nodes = copy.deepcopy(node_list)
    m_long, n_long = 3, nodes[0].shape[1]
    m_short, n_short = 3, nodes[1].shape[1]

    for j in range(m_long):
        for i in range(1, n_long):
            end = j * 48 + 3 * i
            xyz = _cell(nodes, 0, j, i)[:3] + x[end - 3 : end]
            _set_cell_xyz(nodes, 0, j, i, xyz)

    for j in range(m_short):
        for i in range(1, n_short):
            end = j * 27 + 3 * i + 48 * 3
            xyz = _cell(nodes, 1, j, i)[:3] + x[end - 3 : end]
            _set_cell_xyz(nodes, 1, j, i, xyz)

    dist = np.zeros(6)
    for j in range(m_long):
        for i in range(n_long - 1):
            dist[j] += np.linalg.norm(_cell(nodes, 0, j, i)[:3] - _cell(nodes, 0, j, i + 1)[:3])

    for j in range(m_short):
        for i in range(n_short - 1):
            dist[j + 3] += np.linalg.norm(_cell(nodes, 1, j, i)[:3] - _cell(nodes, 1, j, i + 1)[:3])
    return dist


def constraint_hz_part(x, node_list):
    x = np.asarray(x, dtype=float).reshape(-1)
    hz = np.zeros((6, x.size))
    m_long, n_long = 3, node_list[0].shape[1]
    m_short, n_short = 3, node_list[1].shape[1]

    for j in range(m_long):
        for i in range(1, n_long):
            end = j * 48 + 3 * i
            current = _cell(node_list, 0, j, i)[:3] + x[end - 3 : end]
            if i == 1:
                previous = _cell(node_list, 0, j, i - 1)[:3]
                delta = current - previous
                hz[j, end - 3 : end] += delta / np.linalg.norm(delta)
            else:
                previous = _cell(node_list, 0, j, i - 1)[:3] + x[end - 6 : end - 3]
                delta = current - previous
                unit = delta / np.linalg.norm(delta)
                hz[j, end - 3 : end] += unit
                hz[j, end - 6 : end - 3] -= unit

    for j in range(m_short):
        for i in range(1, n_short):
            end = j * 27 + 3 * i + 48 * 3
            current = _cell(node_list, 1, j, i)[:3] + x[end - 3 : end]
            if i == 1:
                previous = _cell(node_list, 1, j, i - 1)[:3]
                delta = current - previous
                hz[j + 3, end - 3 : end] += delta / np.linalg.norm(delta)
            else:
                previous = _cell(node_list, 1, j, i - 1)[:3] + x[end - 6 : end - 3]
                delta = current - previous
                unit = delta / np.linalg.norm(delta)
                hz[j + 3, end - 3 : end] += unit
                hz[j + 3, end - 6 : end - 3] -= unit

    return hz
