from fastapi import FastAPI
from sickbay import to_celsius

app = FastAPI()

@app.get("/")
def read_root():
	return {"Hello" : "World"}

@app.get("/hello/{name}")
def call_name(name: str = "placeholder"):
	return {"message" : f"Quack {name}! Dr. Honk will see you now."}

@app.get("/fever")
def convert_to_C(temperature: float, unit: str | None = None):
	fever = False
	cels = to_celsius(temperature, unit)
	if cels >= 38:
		fever=True
	return {"celsius": cels, "°C, fever": str(fever).lower()}