from solar_energy_forecast.score import score, dummy_predict, zero_predict


if __name__ == "__main__":
    print(score(dummy_predict))
    print(score(zero_predict))
