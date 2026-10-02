# Data dictionary

The raw dataset contains vehicle attributes, service history, condition indicators, and the binary `Need_Maintenance` target.

Derived fields include:

- `days_since_last_service`: days from the configured reference date to the last service.
- `days_to_warranty_end`: days from the configured reference date to warranty expiry.
- `Tire_num`, `Brake_num`, `Battery_num`: ordinal condition scores.
- `History_num`: maintenance-history ordinal score.
- `condition_score`: mean of tire, brake, and battery scores.

Categorical vehicle, fuel, transmission, and owner fields are one-hot encoded.
