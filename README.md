# go-enseirb-2020

Reimplementation of AlphaGoZero [1].


## Files

This branch is used for a variant of the AlphaGo Zero.
The network used only output the probabilities of the moves.
And then we build MCTS on top of those probabilitites but without any rollouts.

The file ```mcts_nn_player.py``` along with the ```model.pt``` can be used to have a working player.
We believe it's quite good but not up to par with the full algorithm.

---

## Bibliography

- **[1]**
  David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, Yutian Chen, Timothy Lillicrap, Fan Hui, Laurent Sifre, George van den Driessche, Thore Graepel & Demis Hassabis  \
  *Mastering the game of Go without human knowledge* \
  19 October 2017, Nature volume 550, pages 354–359 \
  [link](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ)
