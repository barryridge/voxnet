import os

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(ROOT_PATH,'scripts')

TRAIN_PY =  os.path.join(SCRIPT_PATH,'train.py')
TEST_PY =  os.path.join(SCRIPT_PATH,'test.py')
VIZORTHO_PY = os.path.join(SCRIPT_PATH,'output_viz_ortho.py')
REPORT_PY  = os.path.join(SCRIPT_PATH,'train_test_reports.py')
