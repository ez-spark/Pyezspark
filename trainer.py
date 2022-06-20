import src.protocol_trainer as trainer

t = trainer.Trainer('CartPole-v0', 'training_public_key1', 4, 2, 4, 120, 3)
t.connect('0.0.0.0', 8080)
t.train()
