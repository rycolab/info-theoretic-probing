LANGUAGE := tamil
TASK := pos_tag
DATA_DIR := ./data
CHECKPOINT_DIR := ./checkpoints

UD_DIR_BASE := $(DATA_DIR)/ud

UDURL := https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3105/ud-treebanks-v2.5.tgz

UD_DIR := $(UD_DIR_BASE)/ud-treebanks-v2.5
UD_FILE := $(UD_DIR_BASE)/ud-treebanks-v2.5.tgz

PROCESSED_DIR_BASE := $(DATA_DIR)/processed/
PROCESSED_DIR := $(PROCESSED_DIR_BASE)/$(LANGUAGE)
PROCESSED_FILE := $(PROCESSED_DIR)/test--bert.pickle.bz2

TRAIN_DIR := $(CHECKPOINT_DIR)/$(TASK)/$(LANGUAGE)
TRAIN_BERT := $(TRAIN_DIR)/bert/finished.txt
TRAIN_FAST := $(TRAIN_DIR)/fast/finished.txt
TRAIN_ONEHOT := $(TRAIN_DIR)/onehot/finished.txt
TRAIN_RANDOM := $(TRAIN_DIR)/random/finished.txt

all: get_ud process train
	echo "Finished everything"

train: $(TRAIN_BERT) $(TRAIN_FAST) $(TRAIN_ONEHOT) $(TRAIN_RANDOM)

train_bert: $(TRAIN_BERT)

train_fast: $(TRAIN_FAST)

train_onehot: $(TRAIN_ONEHOT)

train_random: $(TRAIN_RANDOM)

process: $(PROCESSED_FILE)

get_ud: $(UD_DIR)

$(TRAIN_RANDOM):
	echo "Train onehot model"
	python -u src/h02_learn/random_search.py --language $(LANGUAGE) --data-path $(PROCESSED_DIR_BASE) --rep 'random' --checkpoint-path $(CHECKPOINT_DIR) --task $(TASK)
# 	python src/h02_learn/train.py --language $(LANGUAGE) --data-path $(PROCESSED_DIR_BASE) --rep 'random' --checkpoint-path $(CHECKPOINT_DIR) --task $(TASK)

$(TRAIN_ONEHOT):
	echo "Train onehot model"
	python -u src/h02_learn/random_search.py --language $(LANGUAGE) --data-path $(PROCESSED_DIR_BASE) --rep 'onehot' --checkpoint-path $(CHECKPOINT_DIR) --task $(TASK)
# 	python src/h02_learn/train.py --language $(LANGUAGE) --data-path $(PROCESSED_DIR_BASE) --rep 'onehot' --checkpoint-path $(CHECKPOINT_DIR) --task $(TASK)

$(TRAIN_FAST):
	echo "Train fasttext model"
	python -u src/h02_learn/random_search.py --language $(LANGUAGE) --data-path $(PROCESSED_DIR_BASE) --rep 'fast' --checkpoint-path $(CHECKPOINT_DIR) --task $(TASK)
# 	python src/h02_learn/train.py --language $(LANGUAGE) --data-path $(PROCESSED_DIR_BASE) --rep 'fast' --checkpoint-path $(CHECKPOINT_DIR) --task $(TASK)

$(TRAIN_BERT):
	echo "Train bert model"
	python -u src/h02_learn/random_search.py --language $(LANGUAGE) --data-path $(PROCESSED_DIR_BASE) --rep 'bert' --checkpoint-path $(CHECKPOINT_DIR) --task $(TASK)
# 	python src/h02_learn/train.py --language $(LANGUAGE) --data-path $(PROCESSED_DIR_BASE) --rep 'bert' --checkpoint-path $(CHECKPOINT_DIR) --task $(TASK)

# Preprocess data
$(PROCESSED_FILE):
	echo "Process language in ud " $(LANGUAGE)
	python src/h01_data/process.py --language $(LANGUAGE) --ud-path $(UD_DIR) --output-path $(PROCESSED_DIR_BASE)

# Get Universal Dependencies data
$(UD_DIR):
	echo "Get ud data"
	mkdir -p $(UD_DIR_BASE)
	wget -P $(UD_DIR_BASE) $(UDURL)
	tar -xvzf $(UD_FILE) -C $(UD_DIR_BASE)
