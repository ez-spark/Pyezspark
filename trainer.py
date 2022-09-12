import src.protocol_trainer as trainer

t = trainer.Trainer('CartPole-v0', '45f855fc760758041904cf09d11169aca357f445e4e49732a9b0bb061e7571ba', 4, 2, 4, 120, 3, username = 'Uzzioo')
t.connect('0.0.0.0', 9080)
t.train()
