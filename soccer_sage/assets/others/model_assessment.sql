WITH raw_data AS (
  SELECT
    id,
    (CASE
  	   WHEN (score_home_team - score_away_team) > 0 THEN 0
  	   WHEN (score_home_team - score_away_team) = 0 THEN 1
  	   WHEN (score_home_team - score_away_team) < 0 THEN 2
     END) AS result,
    tf_prediction,
    league,
    (CASE
       WHEN tf_prediction = 0 THEN (betplay_home_wins - 1)
       WHEN tf_prediction = 1 THEN (betplay_draw - 1)
       WHEN tf_prediction = 2 THEN (betplay_away_wins - 1)
     END) AS possible_earnings,
    DATE(match_date) AS match_date
  FROM matches
  WHERE tf_prediction IS NOT NULL
  AND match_date < DATE(NOW())
  AND betplay_home_wins IS NOT NULL
  AND match_date > '2019-06-01'
  AND tf_prediction != 1

)

SELECT
  tf_prediction,
  -- league,
  -- match_date,
  SUM(CASE
    WHEN result = tf_prediction THEN 1
    ELSE 0
  END)/COUNT(*) AS accuracy,
  SUM(CASE
    WHEN result = tf_prediction THEN 1
    ELSE 0
  END) AS good_guess,
  SUM(CASE
    WHEN result != tf_prediction THEN 1
    ELSE 0
  END) AS bad_guess,
  COUNT(*) AS total,
  SUM(CASE
    WHEN result = tf_prediction THEN possible_earnings
    ELSE -1
  END) AS profit,
  SUM(CASE
    WHEN result = tf_prediction THEN possible_earnings
    ELSE -1
  END)/COUNT(*) AS roi
FROM raw_data
-- GROUP BY match_date
-- GROUP BY league
GROUP BY tf_prediction
HAVING total > 40
-- Optimizing for ROI
ORDER BY roi DESC;

SELECT
  DATE(match_date) AS matchdate,
  COUNT(*) AS num_matches
FROM matches
WHERE league IN ('uefa.euroq', 'ned.1', 'eng.1', 'por.1', 'ned.2')
GROUP BY matchdate
ORDER BY matchdate DESC
LIMIT 200;

SELECT
  match_date,
  team_home.name AS home_team,
  team_away.name AS away_team,
  (CASE
    WHEN tf_prediction = 0 THEN 'Home Wins'
    WHEN tf_prediction = 1 THEN 'Draw'
    WHEN tf_prediction = 2 THEN 'Away Wins'
   END) AS bet
FROM matches
JOIN teams AS team_home ON team_home.id = matches.id_home_team
JOIN teams AS team_away ON team_away.id = matches.id_away_team
WHERE league IN ('mex.1', 'arg.3', 'col.2', 'bra.1', 'per.1', 'bol.1', 'jpn.1' 'bra.2', 'chi.1', 'ned.1')
AND tf_prediction IS NOT NULL
ORDER BY match_date DESC;

SELECT
  match_date,
  team_home.name AS home_team,
  team_away.name AS away_team,
  (CASE
    WHEN tf_prediction = 0 THEN 'Home Wins'
    WHEN tf_prediction = 1 THEN 'Draw'
    WHEN tf_prediction = 2 THEN 'Away Wins'
   END) AS bet
FROM matches
JOIN teams AS team_home ON team_home.id = matches.id_home_team
JOIN teams AS team_away ON team_away.id = matches.id_away_team
WHERE tf_confidence >=0.7
ORDER BY match_date DESC;

WITH raw_data AS (
  SELECT
    (CASE 
       WHEN score_home_team + score_away_team < betplay_threshold THEN 'Under'
       WHEN score_home_team + score_away_team > betplay_threshold THEN 'Over'
     END) AS result,
    (CASE tf_total_goals
       WHEN 0 THEN 'Under'
       WHEN 1 THEN 'Over'
     END) AS prediction,
    (CASE tf_total_goals
       WHEN 0 THEN (betplay_under - 1)
       WHEN 1 THEN (betplay_over - 1)
     END) AS possible_earnings,
    tf_tg_confidence AS tf_confidence,
    DATE(match_date) AS match_date
  FROM matches
  WHERE tf_total_goals IS NOT NULL
  AND match_date < '2019-06-29'
  AND tf_tg_confidence > 0.6
)

SELECT
  match_date,
  COUNT(*) AS total,
  SUM(CASE
    WHEN result = prediction THEN 1
    ELSE 0
  END)/COUNT(*) AS accuracy,
  SUM(CASE
    WHEN result = prediction THEN possible_earnings
    ELSE -1
  END) AS profit,
  SUM(CASE
    WHEN result = prediction THEN possible_earnings
    ELSE -1
  END)/COUNT(*) AS roi
FROM raw_data
GROUP BY match_date
ORDER BY roi DESC;


SELECT
  match_date,
  team_home.name AS home_team,
  team_away.name AS away_team,
  (CASE tf_total_goals
       WHEN 0 THEN 'Under'
       WHEN 1 THEN 'Over'
   END) AS bet
FROM matches
JOIN teams AS team_home ON team_home.id = matches.id_home_team
JOIN teams AS team_away ON team_away.id = matches.id_away_team
WHERE tf_tg_confidence >=0.6
ORDER BY match_date DESC;


SELECT
  match_date,
  league,
  team_home.name AS home_team,
  team_away.name AS away_team,
  (CASE
    WHEN tf_prediction_test = 0 THEN 'Home Wins'
    WHEN tf_prediction_test = 1 THEN 'Draw'
    WHEN tf_prediction_test = 2 THEN 'Away Wins'
   END) AS bet,
   (CASE
       WHEN (tf_prediction_test = 0
             AND betplay_home_wins <= betplay_draw
             AND betplay_home_wins <= betplay_away_wins) THEN 1
       WHEN (tf_prediction_test = 1
             AND betplay_draw <= betplay_home_wins
             AND betplay_draw <= betplay_away_wins) THEN 1
       WHEN (tf_prediction_test = 2
             AND betplay_away_wins <= betplay_home_wins
             AND betplay_away_wins <= betplay_draw) THEN 1
       ELSE 0
   END) AS betplay_approves
FROM matches
JOIN teams AS team_home ON team_home.id = matches.id_home_team
JOIN teams AS team_away ON team_away.id = matches.id_away_team
WHERE tf_confidence_test > 0.5
AND betplay_draw IS NOT NULL
HAVING betplay_approves = 1
ORDER BY match_date DESC;


