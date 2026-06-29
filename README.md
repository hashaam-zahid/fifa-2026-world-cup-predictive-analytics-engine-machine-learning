# <p align="center"><img src="https://filegoat.s3.de.io.cloud.ovh.net/413cfb87-969b-4cc8-be15-981b1359279a/fifa-world-cup-2026-logo-vector-60988539_0.jpg" alt="FIFA 2026 Logo" width="150" height="250"></p>

<h3 align="center">FIFA World Cup 2026 Live Match Predictor Engine</h3>

<p align="center"><img src="https://filegoat.s3.de.io.cloud.ovh.net/2dc2a4e1-bf63-4891-bbd1-422b27fd8dd1/Gemini_Generated_Image_20qm5s20qm5s20qm_0.png" alt="FIFA Predictor Banner" width="100%"></p>

This repository hosts an automated, end-to-end data pipeline script built in Python. The system converges real-time web data acquisition with predictive analytics to output live match-day win/draw probabilities and sleek distribution graphs.

---

## Technical Architecture Overview

The repository is structured into three distinct architectural layers running sequentially in a unified script execution:

*   **Data Acquisition Layer (Web Scraping):** Initiates an on-demand crawl using `BeautifulSoup4` and `requests`. It bypasses static limits with custom headers to parse live national team point coefficients directly from active global football indices (`FotMob`). 
*   **Analytics Optimization Engine:** Handles data transformations via `NumPy` and `Pandas`. It resolves structural strings into floats, manages runtime exceptions, and feeds data directly into the predictive model.
*   **Visualization Interface:** Feeds the statistical array into `Matplotlib` to render clean, publication-ready probability bar charts with dynamic text badges overlaid on the graphics.

---

## Predictive Model & Mathematics Overview

The analytical core of this engine relies on a **Modified Elo Rating System & Normalized Logistic Distribution Framework**. This algorithm is widely utilized in sports analytics to map competitive parity and predict match outcomes based on historical strength indices.

### 1. Expected Outcome Formula
The algorithm evaluates the strength difference between Team A and Team B, passing the delta through a base-10 logistic function. This ensures performance gaps alter win shares non-linearly (a 100-point gap matters more at the top of the table than at the bottom).

$$P(\text{Team 1}_{\text{raw}}) = \frac{1}{1 + 10^{\frac{\text{Rating}_{\text{Team 2}} - \text{Rating}_{\text{Team 1}}}{400}}}$$

### 2. Dynamic Draw Scaling & Normalization
Because tournament formats allow for group-stage draws, a secondary algorithm allocates a draw threshold based on team parity. The closer the raw probabilities are to each other ($0.5$ vs $0.5$), the higher the likelihood of a draw. 

Once the draw probability is carved out, the remaining probability space is proportionally divided between the teams to guarantee the system obeys the probability axiom:

$$P(\text{Team 1}) + P(\text{Draw}) + P(\text{Team 2}) = 1.0 \ (100\%)$$

---

## Model Code Preview

Below is the core algorithmic block extracted from the main script showing exactly how the logistic distribution difference, dynamic draw margin, and probability normalizations are computed:

```python
# MATHEMATICAL PREDICTIVE ALGORITHM (EXTRACT)

# 1. Calculate basic Elo logistic distribution difference
rating_diff = (pts_team_2 - pts_team_1) / 400.0
prob_1_raw = 1.0 / (1.0 + np.power(10.0, rating_diff))
prob_2_raw = 1.0 - prob_1_raw

# 2. Factor in dynamic draw margin criteria based on team parity
draw_prob = 0.26 * (1.0 - abs(prob_1_raw - prob_2_raw))

# 3. Normalize the final outcome array to ensure total sums to exactly 1.0
remaining_prob = 1.0 - draw_prob
team_1_prob = prob_1_raw * remaining_prob
team_2_prob = prob_2_raw * remaining_prob

# 4. Diagnostics Output
total_prob = team_1_prob + draw_prob + team_2_prob
print(f"Team 1 Win Share: {team_1_prob*100:.2f}%")
print(f"Draw Share: {draw_prob*100:.2f}%")
print(f"Team 2 Win Share: {team_2_prob*100:.2f}%")
``` 

### Developer and Credits

This predictive data pipeline was designed, engineered, and optimized by:

*   **Lead Developer:** Hashaam Zahid
*   **Contact/Inquiries:** [hashaamzahid3@gmail.com](mailto:hashaamzahid3@gmail.com)

*Feel free to reach out for collaborations, feature integrations, or questions regarding the predictive algorithms used in this architecture.* 

