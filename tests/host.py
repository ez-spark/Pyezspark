import pyezspark
training_public_key = 'f639be4c66eb938971c296c9140e1f9a6dd3255f80d04c7a94c646422a57dc21'
training_private_key = '06360352c7ad60c25f1a82fc00913f6b33c001a06ff99ab0c72e20419db565b6'
ez = pyezspark.EzSpark(training_public_key, training_private_key = training_private_key)
ez.execute()
