from typing import List

from fastapi import HTTPException
from pydantic import BaseModel, field_validator

VALID_AIRLINES = [
    "Grupo LATAM",
    "Sky Airline",
    "Aerolineas Argentinas",
    "Copa Air",
    "Latin American Wings",
    "Avianca",
    "JetSmart SPA",
    "Gol Trans",
    "American Airlines",
    "Air Canada",
    "Iberia",
    "Delta Air",
    "Air France",
    "Aeromexico",
    "United Airlines",
    "Oceanair Linhas Aereas",
    "Alitalia",
    "K.L.M.",
    "British Airways",
    "Qantas Airways",
    "Lacsa",
    "Austral",
    "Plus Ultra Lineas Aereas"
]


class FlightTemplate(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

    @field_validator("OPERA")
    def validate_airline(cls, operator):
        if operator not in VALID_AIRLINES:
            raise HTTPException(status_code=400, detail='Invalid OPERA.')
        return operator

    @field_validator("TIPOVUELO")
    def validate_type(cls, flight_type):
        if flight_type not in ["N", "I"]:
            raise HTTPException(status_code=400, detail='Invalid TIPOVUELO. Must be N or I.')
        return flight_type

    @field_validator("MES")
    def validate_month(cls, month):
        if month < 1 or month > 12:
            raise HTTPException(status_code=400, detail='Invalid MES. Must be between 1 and 12.')
        return month


class RequestTemplate(BaseModel):
    flights: List[FlightTemplate]
