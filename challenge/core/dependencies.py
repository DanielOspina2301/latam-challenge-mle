from fastapi import Request
from challenge.model import DelayModel


def get_model(request: Request) -> DelayModel:
    return request.app.state.model