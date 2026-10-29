"""One learning-rate intervention; preserve v7 execution and all its gates."""
from copy import deepcopy

from .intrinsic_memory_exposure_v7 import (
    FORMAT as PREVIOUS_FORMAT, TRAINING as PREVIOUS_TRAINING,
    validate_protocol as validate_previous,
)

FORMAT = 'replica.intrinsic_memory.smaller_step.v9'
PACKET = 'replica.intrinsic_memory.launch_packet.v9'
AUTHORIZATION = 'replica.intrinsic_memory.external_authorization.v9'
TRAINING = {**PREVIOUS_TRAINING, 'lr': 0.0001}


def validate_protocol(protocol):
    if protocol.get('format') != FORMAT or protocol.get('training') != TRAINING:
        raise ValueError('fixed smaller-step protocol differs')
    common = deepcopy(protocol)
    common['format'] = PREVIOUS_FORMAT
    common['training'] = deepcopy(PREVIOUS_TRAINING)
    validate_previous(common)
