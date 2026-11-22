"""Create a compact, weights-only copy of the V3-SE3-full epoch20 checkpoint."""

import argparse
import os
import tempfile
from collections import OrderedDict

import torch


FORBIDDEN_PREFIXES = ('planner_head.', 'planner_offset_head.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-keys', type=int, default=1726)
    args = parser.parse_args()

    source = os.path.abspath(args.input)
    destination = os.path.abspath(args.output)
    if source == destination:
        raise ValueError('input and output checkpoint paths must differ')
    if not os.path.isfile(source):
        raise FileNotFoundError(source)

    checkpoint = torch.load(source, map_location='cpu')
    source_state = checkpoint.get('state_dict', checkpoint)
    if not isinstance(source_state, (dict, OrderedDict)):
        raise TypeError('checkpoint state_dict must be a mapping')
    if len(source_state) != args.expected_keys:
        raise AssertionError(
            f'expected {args.expected_keys} frontend keys, got '
            f'{len(source_state)}')

    forbidden = [
        key for key in source_state
        if key.startswith(FORBIDDEN_PREFIXES)
    ]
    if forbidden:
        raise AssertionError(
            f'source is not the no-Planner Stage-1 checkpoint: {forbidden[:8]}')

    state_dict = OrderedDict(source_state.items())
    if hasattr(source_state, '_metadata'):
        state_dict._metadata = source_state._metadata

    output = {
        'state_dict': state_dict,
        'source_checkpoint': source,
        'source_epoch': checkpoint.get('epoch'),
        'source_global_iter': checkpoint.get('global_iter'),
        'weights_only': True,
    }
    output_dir = os.path.dirname(destination)
    os.makedirs(output_dir, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix='.frontend-', suffix='.pth', dir=output_dir)
    os.close(handle)
    try:
        torch.save(output, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    print(
        f'prepared {len(state_dict)} frozen frontend keys: '
        f'{source} -> {destination}')


if __name__ == '__main__':
    main()
