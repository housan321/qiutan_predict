
leagues = {}
base_url = 'http://www.football-data.co.uk/mmz4281/'
seasons = ['1819', '1718', '1617', '1516', '1415', '1314']

for league in leagues:
	if leagues[league]['hasSeasons']:
		for season in seasons:
			url = '{0}{1}/{2}.csv'.format(base_url, season,
				                          leagues[league]['slug'])
	else:
		url = leagues[league]['url']
