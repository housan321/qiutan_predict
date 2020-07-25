SELECT
  DISTINCT
  league,
  season,
  bs_num_id AS match_id,
  hometeam AS id_home_team,
  awayteam AS id_away_team,
  bs_time AS match_date,
  (CASE
    WHEN FTR = 'H' THEN 0
    WHEN FTR = 'D' THEN 1
    WHEN FTR = 'A' THEN 2
  END) AS result,
  (CAST(SUBSTRING(res_score, 1, 1) AS SIGNED integer)) AS score_home_team,
  (CAST(SUBSTRING(res_score, 4, 1)  AS SIGNED integer)) AS score_away_team,
  VTFormPtsStr,
  HTFormPtsStr,
  ATFormPtsStr
FROM all_match_samples
WHERE (league = %s AND season = %s)  
ORDER BY match_date ASC;

