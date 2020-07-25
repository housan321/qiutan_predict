WITH teams_with_more_than_n_games AS (
  SELECT
    -- For each team, count the number of games that
    -- exist in the dataset and filter those that
    -- have more or equal than N games
    team_id,
    SUM(n_matches) AS n_matches
  FROM (
    -- Merge teams and game count from home and away
    -- teams
    SELECT
      -- For each home team, count the number of
      -- games that exist in the dataset
      id_home_team AS team_id,
      COUNT(*) AS n_matches
    FROM matches
    GROUP BY team_id

    UNION

    SELECT
      -- For each away team, count the number of
      -- games that exist in the dataset
      id_away_team AS team_id,
      COUNT(*) AS n_matches
    FROM matches
    GROUP BY team_id
  ) AS merged_game_count
  GROUP BY team_id
  HAVING n_matches >= %s
), teams_with_stats AS (
  SELECT
    -- DISTINCT id_team
    id_team, COUNT(*) AS num_stats
  FROM match_stats
  GROUP BY id_team
  HAVING num_stats >= 5
)

SELECT
  -- For each match, get the basic information
  -- 0: Win for home
  -- 1: Draw
  -- 2: Lose for home
  DISTINCT
  matches.id AS id,
  id_home_team,
  id_away_team,
  IF(leagues.id IS NOT NULL, leagues.id, 0) AS league,
  score_home_team,
  score_away_team,
  (CASE
    WHEN (score_home_team - score_away_team) > 0 THEN 0
    WHEN (score_home_team - score_away_team) = 0 THEN 1
    WHEN (score_home_team - score_away_team) < 0 THEN 2
  END) AS result,
  IFNULL(ms_home.fouls, -48.0) AS home_fouls,
  IFNULL(ms_home.yellow_cards, -48.0) AS home_yellow_cards,
  IFNULL(ms_home.red_cards, -48.0) AS home_red_cards,
  IFNULL(ms_home.offsides, -48.0) AS home_offsides,
  IFNULL(ms_home.corners, -48.0) AS home_corners,
  IFNULL(ms_home.saves, -48.0) AS home_saves,
  IFNULL(ms_home.possession, -48.0) AS home_possession,
  IFNULL(ms_home.shots, -48.0) AS home_shots,
  IFNULL(ms_home.shots_on_goal, -48.0) AS home_shots_on_goal,
  IFNULL(ms_away.fouls, -48.0) AS away_fouls,
  IFNULL(ms_away.yellow_cards, -48.0) AS away_yellow_cards,
  IFNULL(ms_away.red_cards, -48.0) AS away_red_cards,
  IFNULL(ms_away.offsides, -48.0) AS away_offsides,
  IFNULL(ms_away.corners, -48.0) AS away_corners,
  IFNULL(ms_away.saves, -48.0) AS away_saves,
  IFNULL(ms_away.possession, -48.0) AS away_possession,
  IFNULL(ms_away.shots, -48.0) AS away_shots,
  IFNULL(ms_away.shots_on_goal, -48.0) AS away_shots_on_goal,
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
-- Litle workaround to work for training and prediction
WHERE (status = "STATUS_FULL_TIME" OR status = %s)
AND match_date <= '2019-05-01'
AND id_home_team IN (
  SELECT
    -- Get teams with more or equal than n games
    team_id
  FROM teams_with_more_than_n_games
)
AND id_home_team IN (
  SELECT
    -- Get teams with stats during at least one match
    id_team
  FROM teams_with_stats
)
AND id_away_team IN (
  SELECT
    -- Get teams with more or equal than n games
    team_id
  FROM teams_with_more_than_n_games
)
AND id_away_team IN (
  SELECT
    -- Get teams with stats during at least one match
    id_team
  FROM teams_with_stats
)
ORDER BY match_date ASC;
