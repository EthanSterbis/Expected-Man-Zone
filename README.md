## Author: Ethan Sterbis | [LinkedIn](https://www.linkedin.com/in/ethansterbis/) | [Twitter](https://x.com/EthanSterbis)

## Introduction
The aim of this model is to predict whether a defense will be in man or zone coverage on passing plays.
The factors of the model are able to be calculated up until the quarterback snaps the ball, as the model is intended to predict pre-snap coverage probability.

This script uses play-by-play data for the 2022 and 2023 NFL seasons, along with participation data as well as data from [FTN](https://www.ftndata.com/).

## Factor Analysis

### Most Important Factors by Gain
<img src="https://github.com/user-attachments/assets/51bb0cdf-1ad3-4ace-b9b3-18a80406a253" width="100%" alt="Defenses Playing Man Coverage" />
The table shown above includes the 10 highest features by gain according to the XGBoost model. Gain represents the average contribution of a feature to reduce the model's error. Also shown are cover, which depicts the number of observations related to a feature, and frequency, which quanitifies the number of times a feature appears in all trees in the model.

## Data Visualization

### 2023 NFL Defenses

#### Defenses Playing Man Coverage
<img src="https://github.com/user-attachments/assets/079cd770-fed4-413a-a4c2-97288a2bae57" width="100%" alt="Defenses Playing Man Coverage" />

#### Defenses Playing Zone Coverage
<img src="https://github.com/user-attachments/assets/05fb701b-d022-4fa0-aa07-b91d3a3677b3" width="100%" alt="Defenses Playing Zone Coverage" />

Defensive takeaways:

For the Denver Broncos, their difference in coverage efficiency can be chalked up to lack of depth at cornerback and EPA volatility due to turnovers.
Based on the model, the Broncos should lean into their proficiency in man coverage to let Patrick Surtain II shine.

The Atlanta Falcons were the highest team in average man coverage over expected at about 10%.
In the zone coverage plot, we can see why. When in man coverage, the Falcons allowed about 0.18 less EPA/play.

The Jacksonville Jaguars and Minnesota Vikings both should have played more zone despite averaging a higher rate than expected.
When playing man coverage, these two teams allowed about 0.43 EPA/play. For reference, when Patrick Mahomes won his 2022 MVP,
he had an EPA/play of 0.30. In other words, the Jaguars and Vikings allowed MVP-level quartback play when in man coverage.
The Vikings, outside of Harrison Smith, did not have a starting defensive back over the age of 25.
Meanwhile, the Jaguars moved on from defensive coordinator Mike Caldwell at the end of the 2023 season despite Caldwell only being there for two years.


### 2023 NFL Offenses

#### Offenses Against Man Coverage
<img src="https://github.com/user-attachments/assets/3d6bdac5-eec5-48ea-ad39-44aed53c21ca" width="100%" alt="Offenses vs. Man Coverage" />

#### Offenses Against Zone Coverage
<img src="https://github.com/user-attachments/assets/28ed74eb-d61b-4040-a76f-11c9071fe082" width="100%" alt="Offenses vs. Zone Coverage" />

Offensive takeaways:

Tampa Bay Buccaneers receiver Mike Evans is a massive man coverage threat (as long as the opposing DB isn't Marshon Lattimore).
The Buccaneers saw the least amount of man coverage over expectation, and ranked 4th in EPA/play against man coverage.

Baltimore Ravens tight end Mark Andrews battled with injury in 2023. Without Andrews,
the Ravens lacked a contested catch threat who could consistently win at the catch point,
resulting in a substantial decrease in efficiency when facing zonman coverage. 



## Notes
It is worth mentioning that data used to draw insights was filtered down to snaps in which there was a pass attempt.
Therefore, EPA was calculated by excluding plays where the defense called man or zone coverage, followed by a rush attempt, penalty, etc.

For any further information and/or inquires, please [reach out to me](https://x.com/EthanSterbis)!

### Mentions: [Tej Seth](https://x.com/tejfbanalytics), [Arjun Menon](https://x.com/arjunmenon100), [Brad Congelio](https://x.com/BradCongelio)
