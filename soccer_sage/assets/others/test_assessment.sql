WITH raw_data AS (
  SELECT
    id,
    (CASE
       WHEN (score_home_team - score_away_team) > 0 THEN 0
       WHEN (score_home_team - score_away_team) = 0 THEN 1
       WHEN (score_home_team - score_away_team) < 0 THEN 2
     END) AS result,
    tf_prediction_test,
    league,
    (CASE
       WHEN tf_prediction_test = 0 THEN (betplay_home_wins - 1)
       WHEN tf_prediction_test = 1 THEN (betplay_draw - 1)
       WHEN tf_prediction_test = 2 THEN (betplay_away_wins - 1)
     END) AS possible_earnings,
    DATE(match_date) AS match_date
  FROM matches
  WHERE tf_prediction_test IS NOT NULL
  AND match_date < DATE(NOW())
  AND DATE(match_date) > '2019-06-01'
  AND betplay_home_wins IS NOT NULL
  AND tf_confidence_test > 0.7
)

SELECT
  -- league,
  -- match_date,
  SUM(CASE
    WHEN result = tf_prediction_test THEN 1
    ELSE 0
  END)/COUNT(*) AS accuracy,
  SUM(CASE
    WHEN result = tf_prediction_test THEN 1
    ELSE 0
  END) AS good_guess,
  SUM(CASE
    WHEN result != tf_prediction_test THEN 1
    ELSE 0
  END) AS bad_guess,
  COUNT(*) AS total,
  SUM(CASE
    WHEN result = tf_prediction_test THEN possible_earnings
    ELSE -1
  END) AS profit,
  SUM(CASE
    WHEN result = tf_prediction_test THEN possible_earnings
    ELSE -1
  END)/COUNT(*) AS roi
FROM raw_data
-- GROUP BY match_date
-- GROUP BY league
HAVING total > 10
-- Optimizing for ROI
ORDER BY roi DESC;

SELECT
  DATE(match_date) AS matchdate,
  COUNT(*) AS num_matches
FROM matches
WHERE league IN ('uefa.euroq', 'esp.2', 'concacaf.gold')
GROUP BY matchdate
ORDER BY matchdate DESC
LIMIT 200;

SELECT
  match_date,
  team_home.name AS home_team,
  team_away.name AS away_team,
  (CASE
    WHEN tf_prediction_test = 0 THEN 'Home Wins'
    WHEN tf_prediction_test = 1 THEN 'Draw'
    WHEN tf_prediction_test = 2 THEN 'Away Wins'
   END) AS bet
FROM matches
JOIN teams AS team_home ON team_home.id = matches.id_home_team
JOIN teams AS team_away ON team_away.id = matches.id_away_team
WHERE league IN ('uefa.euroq', 'esp.2', 'concacaf.gold')
AND tf_prediction_test IS NOT NULL
ORDER BY match_date DESC
LIMIT 100;