# Pyezspark

# Install With PIP

```
pip install Pyezspark
```

# Install From the repo

- Download this repo

- run in the repo directory

```
sh install.sh
```

# Do You want to train other people models?

- Go to https://app.ezspark.ai

- Select a training from the dashboard

- In the details take the training public key

- Run

```
import pyezspark
training_public_key = ''
ez = pyezspark.EzSpark(training_public_key)
ez.execute()
```

# Do You want to host a training?

- Go to https://app.ezspark.ai

- Create an account

- Create a new training

- Get the training public key and training private key from the info of "My Trainings"

- Run

```
import pyezspark
training_public_key = ''
training_private_key = ''
ez = pyezspark.EzSpark(training_public_key, training_private_key = training_private_key)
ez.execute()
```
