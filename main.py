from src.data_prep import main as prep_main
from src.modeling import main as model_main
from src.scoring import main as score_main

if __name__ == '__main__':
    prep_main()
    model_main()
    score_main()