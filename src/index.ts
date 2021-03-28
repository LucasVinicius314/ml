import * as tf from '@tensorflow/tfjs'

type Row = {
  '': number
  customerID: string
  gender: 'Female' | 'Male'
  SeniorCitizen: 0 | 1
  Partner: 'No' | 'Yes'
  Dependents: 'No' | 'Yes'
  tenure: number
  PhoneService: 'No' | 'Yes'
  MultipleLines: string
  InternetService: string
  OnlineSecurity: string
  OnlineBackup: string
  DeviceProtection: string
  TechSupport: string
  StreamingTV: string
  StreamingMovies: string
  Contract: string
  PaperlessBilling: 'Yes' | 'No'
  PaymentMethod: string
  MonthlyCharges: number
  TotalCharges: number
  Churn: 'Yes' | 'No'
}

const gender = {
  Male: 0,
  Female: 1,
  0: 'Male',
  1: 'Female',
}

const binary = {
  No: 0,
  Yes: 1,
  0: 'No',
  1: 'Yes',
}

const main = async () => {
  // shape data

  const data = (await tf.data
    .csv('file://D:/GitHub/ml/data/telecom_users.csv')
    .toArray()) as Row[]

  const features = data.map((v) => v.TotalCharges / 999)
  const labels = data.map((v) => binary[v.Churn])

  // const max = features.sort()[features.length - 1]

  // console.log(max)

  const xs = tf.tensor2d(features, [features.length, 1])
  const ys = tf.tensor2d(labels, [labels.length, 1])

  // define model

  const model = tf.sequential()
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }))

  model.compile({
    loss: tf.losses.meanSquaredError,
    optimizer: tf.train.adam(0.1),
    metrics: ['accuracy'],
  })

  // fit model

  await model.fit(xs, ys, {
    epochs: 100,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch}: ${logs.loss}`)
      },
    },
  })

  // predict

  const predictData = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    100,
    300,
    600,
    1000,
    1300,
    1600,
    1900,
    2500,
    3000,
    3500,
    4000,
    4500,
    5000,
    6000,
    7000,
    8000,
    9000,
  ].map((v) => v / 999)

  const predictTensor = tf.tensor2d(predictData, [predictData.length, 1])

  const predict = model.predict(predictTensor) as tf.Tensor
  const res = predict.dataSync()
  console.log(res)
  console.log(Array.from(res).map((v) => binary[Math.round(v)]))
}

main()
