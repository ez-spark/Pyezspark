# Pyezspark

# Do You want to train other people models?

- Go to https://app.ezspark.ai

- Select a training from the dashboard

- In the details take the training public key

- Run

```
from Pyezspark import ezspark
training_public_key = ''
ez = ezspark.EzSpark(training_public_key)
ez.execute()
```

# Do You want to host a training?

- Go to https://app.ezspark.ai

- Create an account

- Create a new training

- Get the training public key and training private key from the info of "My Trainings"

- Run

```
from Pyezspark import ezspark
training_public_key = ''
training_private_key = ''
ez = ezspark.EzSpark(training_public_key, training_private_key = training_private_key)
ez.execute()
```
