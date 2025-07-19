# Communicating the Forecast Method
* Related File: "mo3_gco_forecast_weibull.py"

### Version 1 : Medium Technical Description (for data-savvy stakeholders)
* We model cumulative gross charge-offs (GCOs) using a parametric Weibull cumulative distribution function (CDF).
* A single Weibull shape and scale parameter is estimated per segment using fully mature vintages (2015–2017).
* For more recent vintages (2019–2024), we calibrate a single “amplitude” factor using observed data through 36 months on book.
* The forecast is generated as:
    * Forecast(t) = Amplitude × 𝐹(𝑡−lag;shape,scale)
    * where 𝐹 is the Weibull CDF and the lag is a 6-month delay before losses begin.
* This setup ensures the forecast:
    * Extends smoothly from observed actuals,
    * Asymptotes reasonably,
    * Differentiates across vintages and segments.

### Version 2 : Simplified Business Explanation (for executive stakeholders)
* We’re using historical loan performance to project future losses.
* Older vintages teach us the shape and timing of how losses typically emerge.
* For newer vintages, we anchor to what we’ve seen so far, then project forward using the learned patterns.
* The result: a clear, credible forecast that starts where actuals end and shows how much more loss we expect.
