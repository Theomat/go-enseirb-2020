# go-enseirb-2020

Reimplementation of AlphaGoZero [1].

---

## Bibliography

- **[1]**
  David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, Yutian Chen, Timothy Lillicrap, Fan Hui, Laurent Sifre, George van den Driessche, Thore Graepel & Demis Hassabis  \
  *Mastering the game of Go without human knowledge* \
  19 October 2017, Nature volume 550, pages 354–359 \
  [link](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ)


## Files and Ideas
 - Firstly, we are inspired by alpha go zero, but because self-play time is VERY HIGH (at least some months on oridnary hardware) we start the network with some knowledge.
 
 - We train the network on the standard json9x9 file handed out by the teacher. We do some data augmentation on this data set and do a first training on this dataset. That's why we call our network AlphaGoOne. We do not include the final .npy file with the dataset because it's too large, but it can be generated using the files `dataset_builder.py` and `augmentate.py`. Our dataset builder uses the GnuGo board evaluation feature (scores) to evaluate the boards. The augmentate script allows to configure which kind of data augmentation should be used.

 - The neural network model follows the AlphaGoZero ideas: We have a two-head policy/value network with Residual Blocks. The `model.py` file contains all implementation details. The loss function is a bit tricky to code, therefore, the code is separated in the `loss.py` file with a small test funcion.

 - After training the network with different augmentation parameters (using the `train_alphazero.py` script), we followed we some iterations of self-play. Most of the code used for that can be found in files: `trainer.py` and `auto_train.py`.

 - After many attempts, different tests for hyperparemeters we noticed that training without using the `swap_color` option in the `augmentate.py` file we would have a very good black-side player, but not very good (not terrible too !) if playing as white. The opposite would happen if we used `swap_color`. That's why our final player uses two models: one that is very good playing as black (`model_black.py`) and one as white (`model_white.py`)

 - Clearly, MCTS code plays a huge role in choosing optimal actions guided by its neural network. The code for the MCTS can be found mosyly in `node.py`, `mcts.py` and `alpha_go_one_player.py`

 - We had different attempts with other datasets, therefore the files `get_pro_json.py`. At the end, our best results were using the default dataset given by the teacher.

 - Finally, our model can win as black vs GnuGo level 10 and as white up until lvl 7. To test you can run for example setting the proper level of GnuGo on `GnuGo.py` file.

   ```python3 namedGame.py alpha_go_one_player.py gnugoPlayer.py```

    In this example, we win as black vs GnuGo level 10 100% of times over 100 games.