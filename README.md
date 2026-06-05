# Traffic Demand Prediction — Gridlock Hackathon 2.0

## Approach Summary

### Key Insight
The test set is entirely day 49. The training set contains day 48 data for the 
same geohashes and timestamps. Day-48 demand at the same (geohash, timestamp) 
is therefore a near-direct lag feature and the strongest single predictor.

### Feature Engineering
1. **Day-48 lag demand** — exact match on geohash + timestamp from day 48 train
2. **Fallback hierarchy** when lag is missing: geohash×timeslot mean → geohash 
   mean → geo4-prefix mean → global timeslot mean
3. **Geohash decoded** to lat/lon using manual base32 decoding
4. **Cyclical time encoding** — sin/cos for hour and 15-min time slot
5. **Geohash × timeslot target encoding** — captures location-specific rush-hour 
   patterns per 15-min window
6. **Peak-hour flag** and lanes × peak interaction
7. Label encoding for RoadType, Weather, LargeVehicles, Landmarks

### Model
LightGBM regressor trained on all available training data (day 48 + day 49).
Hyperparameters: 3000 trees, lr=0.03, num_leaves=127, L1/L2 regularization.

### Evaluation
Metric: `max(0, 100 * R²(actual, predicted))`