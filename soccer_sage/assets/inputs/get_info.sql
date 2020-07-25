SELECT
  -- For each match, get the basic information
  -- 0: Win for home
  -- 1: Draw
  -- 2: Lose for home
  DISTINCT
  matches.id AS id,
  id_home_team,
  id_away_team,
  IFNULL(leagues.association, 0) AS league,
  score_home_team,
  score_away_team,
  (CASE
    WHEN (score_home_team - score_away_team) > 0 THEN 0
    WHEN (score_home_team - score_away_team) = 0 THEN 1
    WHEN (score_home_team - score_away_team) < 0 THEN 2
  END) AS result,
  (score_home_team + score_away_team) AS num_goals,
  betplay_threshold,
  IFNULL(ms_home.fouls, -16.0) AS home_fouls,
  IFNULL(ms_home.yellow_cards, -16.0) AS home_yellow_cards,
  IFNULL(ms_home.red_cards, -16.0) AS home_red_cards,
  IFNULL(ms_home.offsides, -16.0) AS home_offsides,
  IFNULL(ms_home.corners, -16.0) AS home_corners,
  IFNULL(ms_home.saves, -16.0) AS home_saves,
  IFNULL(ms_home.possession / 100.0, -16.0) AS home_possession,
  IFNULL(ms_home.shots, -16.0) AS home_shots,
  IFNULL(ms_home.shots_on_goal, -16.0) AS home_shots_on_goal,
  IFNULL(ms_away.fouls, -16.0) AS away_fouls,
  IFNULL(ms_away.yellow_cards, -16.0) AS away_yellow_cards,
  IFNULL(ms_away.red_cards, -16.0) AS away_red_cards,
  IFNULL(ms_away.offsides, -16.0) AS away_offsides,
  IFNULL(ms_away.corners, -16.0) AS away_corners,
  IFNULL(ms_away.saves, -16.0) AS away_saves,
  IFNULL(ms_away.possession / 100.0, -16.0) AS away_possession,
  IFNULL(ms_away.shots, -16.0) AS away_shots,
  IFNULL(ms_away.shots_on_goal, -16.0) AS away_shots_on_goal,
  match_date
FROM matches
LEFT JOIN leagues ON matches.league = leagues.name
LEFT JOIN match_stats AS ms_home ON (
  matches.id = ms_home.id_match
  AND ms_home.id_team = id_home_team
)
LEFT JOIN match_stats AS ms_away ON (
  matches.id = ms_away.id_match
  AND ms_away.id_team = id_away_team
)
ORDER BY match_date ASC;
