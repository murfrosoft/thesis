default:
	python3 figure_creator.py logfiles/kochSnowflake_100_aggregateLog.csv -v

temp:
	python3 bca.py test_images/Larger/norwaymap.png -g 5
	python3 bca.py test_images/Larger/norwaymap.png -g 6
	python3 bca.py test_images/Larger/norwaymap.png -g 7
	python3 bca.py test_images/Larger/norwaymap.png -g 8
	python3 bca.py test_images/Larger/norwaymap.png -g 9
	python3 bca.py test_images/Larger/norwaymap.png -g 10

aggregateU:
	python3 aggregator.py -u logfiles/UniformLogs/blank/*
	python3 aggregator.py -u logfiles/UniformLogs/line/*
	python3 aggregator.py -u logfiles/UniformLogs/circle/*
	python3 aggregator.py -u logfiles/UniformLogs/checkers/*
	python3 aggregator.py -u logfiles/UniformLogs/canopy/*
	python3 aggregator.py -u logfiles/UniformLogs/fifty50/*
	python3 aggregator.py -u logfiles/UniformLogs/kochSnowflake/*
	python3 aggregator.py -u logfiles/UniformLogs/koch3lines/*
	python3 aggregator.py -u logfiles/UniformLogs/fallleaf_edge/5fallleaf/*
	python3 aggregator.py -u logfiles/UniformLogs/fallleaf_edge/10fallleaf/*
	python3 aggregator.py -u logfiles/UniformLogs/fallleaf_edge/15fallleaf/*
	python3 aggregator.py -u logfiles/UniformLogs/fallleaf_edge/20fallleaf/*
	python3 aggregator.py -u logfiles/UniformLogs/fallleaf_edge/25fallleaf/*
	python3 aggregator.py -u logfiles/UniformLogs/fallleaf_edge/30fallleaf/*
	python3 aggregator.py -u logfiles/UniformLogs/fallleaf_edge/35fallleaf/*
	python3 aggregator.py -u logfiles/UniformLogs/fallleaf_edge/40fallleaf/*
	python3 aggregator.py -u logfiles/UniformLogs/fallleaf_edge/45fallleaf/*
	python3 aggregator.py -u logfiles/UniformLogs/fallleaf_edge/50fallleaf/*
	python3 aggregator.py -u logfiles/UniformLogs/fallleaf_edge/55fallleaf/*
	python3 aggregator.py -u logfiles/UniformLogs/fallleaf_edge/60fallleaf/*
	python3 aggregator.py -u logfiles/UniformLogs/fallleaf_edge/65fallleaf/*
	python3 aggregator.py -u logfiles/UniformLogs/fallleaf_edge/70fallleaf/*
	python3 aggregator.py -u logfiles/UniformLogs/fallleaf_edge/75fallleaf/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_edge/5owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_edge/10owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_edge/15owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_edge/20owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_edge/25owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_edge/30owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_edge/35owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_edge/40owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_edge/45owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_edge/50owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_edge/55owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_edge/60owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_edge/65owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_edge/70owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_edge/75owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_thresh/10_thr_owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_thresh/20_thr_owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_thresh/30_thr_owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_thresh/40_thr_owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_thresh/50_thr_owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_thresh/60_thr_owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_thresh/70_thr_owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_thresh/80_thr_owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_thresh/90_thr_owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_thresh/100_thr_owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_thresh/110_thr_owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_thresh/120_thr_owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_thresh/130_thr_owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_thresh/140_thr_owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_thresh/150_thr_owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_thresh/160_thr_owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_thresh/170_thr_owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_thresh/180_thr_owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_thresh/190_thr_owl/*
	python3 aggregator.py -u logfiles/UniformLogs/owl_thresh/200_thr_owl/*

aggregateG:
	python3 aggregator.py -g logfiles/GaussianLogs/line/*
	python3 aggregator.py -g logfiles/GaussianLogs/circle/*
	python3 aggregator.py -g logfiles/GaussianLogs/checkers/*
	python3 aggregator.py -g logfiles/GaussianLogs/canopy/*
	python3 aggregator.py -g logfiles/GaussianLogs/fifty50/*
	python3 aggregator.py -g logfiles/GaussianLogs/kochSnowflake/*
	python3 aggregator.py -g logfiles/GaussianLogs/koch3lines/*
	python3 aggregator.py -g logfiles/GaussianLogs/fallleaf_edge/5fallleaf/*
	python3 aggregator.py -g logfiles/GaussianLogs/fallleaf_edge/10fallleaf/*
	python3 aggregator.py -g logfiles/GaussianLogs/fallleaf_edge/15fallleaf/*
	python3 aggregator.py -g logfiles/GaussianLogs/fallleaf_edge/20fallleaf/*
	python3 aggregator.py -g logfiles/GaussianLogs/fallleaf_edge/25fallleaf/*
	python3 aggregator.py -g logfiles/GaussianLogs/fallleaf_edge/30fallleaf/*
	python3 aggregator.py -g logfiles/GaussianLogs/fallleaf_edge/35fallleaf/*
	python3 aggregator.py -g logfiles/GaussianLogs/fallleaf_edge/40fallleaf/*
	python3 aggregator.py -g logfiles/GaussianLogs/fallleaf_edge/45fallleaf/*
	python3 aggregator.py -g logfiles/GaussianLogs/fallleaf_edge/50fallleaf/*
	python3 aggregator.py -g logfiles/GaussianLogs/fallleaf_edge/55fallleaf/*
	python3 aggregator.py -g logfiles/GaussianLogs/fallleaf_edge/60fallleaf/*
	python3 aggregator.py -g logfiles/GaussianLogs/fallleaf_edge/65fallleaf/*
	python3 aggregator.py -g logfiles/GaussianLogs/fallleaf_edge/70fallleaf/*
	python3 aggregator.py -g logfiles/GaussianLogs/fallleaf_edge/75fallleaf/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_edge/5owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_edge/10owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_edge/15owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_edge/20owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_edge/25owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_edge/30owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_edge/35owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_edge/40owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_edge/45owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_edge/50owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_edge/55owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_edge/60owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_edge/65owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_edge/70owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_edge/75owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_thresh/10_thr_owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_thresh/20_thr_owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_thresh/30_thr_owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_thresh/40_thr_owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_thresh/50_thr_owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_thresh/60_thr_owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_thresh/70_thr_owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_thresh/80_thr_owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_thresh/90_thr_owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_thresh/100_thr_owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_thresh/110_thr_owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_thresh/120_thr_owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_thresh/130_thr_owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_thresh/140_thr_owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_thresh/150_thr_owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_thresh/160_thr_owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_thresh/170_thr_owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_thresh/180_thr_owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_thresh/190_thr_owl/*
	python3 aggregator.py -g logfiles/GaussianLogs/owl_thresh/200_thr_owl/*

ufigs:
	python3 figure_creator.py -au logfiles/aggregLogs/UniformAglog/blank_u100_aglog.csv
	python3 figure_creator.py -au logfiles/aggregLogs/UniformAglog/canopy_u100_aglog.csv
	python3 figure_creator.py -au logfiles/aggregLogs/UniformAglog/circle_u100_aglog.csv
	python3 figure_creator.py -au logfiles/aggregLogs/UniformAglog/fifty50_u100_aglog.csv
	python3 figure_creator.py -au logfiles/aggregLogs/UniformAglog/koch3lines_u100_aglog.csv
	python3 figure_creator.py -au logfiles/aggregLogs/UniformAglog/kochSnowflake_u100_aglog.csv
	python3 figure_creator.py -au logfiles/aggregLogs/UniformAglog/line_u100_aglog.csv
	python3 figure_creator.py -au logfiles/aggregLogs/UniformAglog/norwaymap_u100_aglog.csv

gfigs:
	python3 figure_creator.py -ag logfiles/aggregLogs/GaussianAglog/canopy_g100_aglog.csv
	python3 figure_creator.py -ag logfiles/aggregLogs/GaussianAglog/circle_g100_aglog.csv
	python3 figure_creator.py -ag logfiles/aggregLogs/GaussianAglog/fifty50_g100_aglog.csv
	python3 figure_creator.py -ag logfiles/aggregLogs/GaussianAglog/koch3lines_g100_aglog.csv
	python3 figure_creator.py -ag logfiles/aggregLogs/GaussianAglog/kochSnowflake_g100_aglog.csv
	python3 figure_creator.py -ag logfiles/aggregLogs/GaussianAglog/line_g100_aglog.csv
	python3 figure_creator.py -ag logfiles/aggregLogs/GaussianAglog/norwaymap_g100_aglog.csv
