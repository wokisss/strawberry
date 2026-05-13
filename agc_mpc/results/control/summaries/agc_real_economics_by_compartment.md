# AGC Real Resource And Economics Baseline

This table estimates compartment-level resource use and approximate economics from the recorded AGC 2019 data. Net profit is approximate: tomato income uses the official date/Brix price table with nearest available TSS measurements, crop maintenance uses recorded stem density, and plant fixed cost uses the documented two-stem plant price.

| compartment | income | variable cost | fixed plant cost | approx net profit | tomato kg/m2 | heat MJ/m2 | elec kWh/m2 | CO2 kg/m2 | irrigation L/m2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Automatoes | 36.24 | 25.90 | 4.29 | 6.05 | 14.92 | 185.3 | 270.4 | 9.07 | 788.9 |
| AICU | 34.74 | 24.92 | 3.96 | 5.85 | 13.76 | 252.3 | 240.3 | 10.15 | 553.5 |
| Reference | 35.33 | 28.65 | 3.08 | 3.60 | 14.30 | 471.6 | 267.2 | 8.63 | 788.6 |
| IUACAAS | 32.82 | 25.24 | 4.29 | 3.29 | 13.48 | 334.8 | 228.0 | 7.28 | 866.6 |
| Digilog | 34.35 | 28.37 | 2.86 | 3.12 | 14.21 | 173.0 | 323.2 | 9.66 | 741.1 |
| TheAutomators | 36.00 | 29.44 | 3.96 | 2.60 | 14.36 | 362.6 | 284.8 | 12.50 | 723.2 |

Official variable-cost rules encoded here:

- electricity: `0.08 EUR/kWh` peak and `0.04 EUR/kWh` off-peak
- heat: `0.0083 EUR/MJ`
- CO2: `0.08 EUR/kg` for the first `12 kg/m2`, then `0.20 EUR/kg`
- crop maintenance: `0.0085 EUR per stem/m2 per day`
- Class A tomatoes use full estimated price; Class B uses half price