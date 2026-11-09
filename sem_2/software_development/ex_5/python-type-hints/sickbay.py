"""Sick bay software of the SS Probably Fine.

Maintained by Dr. Honk, who does not believe in type hints. Yet.

Temperatures are measured in degrees Celsius, unless stated otherwise.
"""

import random
from typing import TypeVar

T = TypeVar("T")

PATIENTS = {
    "Captain Vega": 37.1,
    "Lt. Byte": 38.6,
    "Cmdr. Nova": 36.8,
    "Ensign Pixel": 39.2,
}

FEVER_THRESHOLD = 38.0


def to_celsius(value: float, unit: str) -> float:
    """Convert a temperature in the given unit ("C", "F", or "K") to Celsius."""
    if unit == "C":
        return value
    if unit == "F":
        return (value - 32) * 5 / 9
    if unit == "K":
        return value - 273.15
    raise ValueError(f"Unknown unit: {unit}")


def has_fever(celsius: float) -> bool:
    return celsius >= FEVER_THRESHOLD


def fever_patients(patients: dict[str, float]) -> list[str]:
    """Return the names of all patients with fever, in alphabetical order."""
    return sorted(name for name, celsius in patients.items() if has_fever(celsius))


def find_patient(query: str, patients: dict[str, float]) -> str:
    """Return the full name of the first patient whose name contains `query`."""
    for name in patients:
        if query.lower() in name.lower():
            return name
    return name
    #return None


def average_temperature(temperatures: list[float])-> float | str:
    if not temperatures:
        return "n/a"
    return sum(temperatures) / len(temperatures)


def pick_random(items: list[T])-> T:
    """Return a random element of a list."""
    return random.choice(items)


def greet(query: str, patients: dict[str, float]) -> str:
    name = find_patient(query, patients)
    return f"Quack, {name.upper()}! Dr. Honk will see you now."


def ward_report(patients: dict[str, float]) -> str:
    average = average_temperature(list(patients.values()))
    if type(average) == float:
        lines = [f"Patients: {len(patients)}", f"Average temperature: {round(average, 1)} °C"]
        for name in fever_patients(patients):
            lines.append(f"  FEVER: {name}")
    else:
        lines = ["average temperature could not be found"]
    return "\n".join(lines)


if __name__ == "__main__":
    print(ward_report(PATIENTS))
    print(greet("byte", PATIENTS))
    print(f"Today's volunteer for the flu shot: {pick_random(list(PATIENTS))}")
    print(f"101.3 °F = {to_celsius(101.3, 'F'):.1f} °C")
