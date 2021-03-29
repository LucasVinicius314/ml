import * as ervy from 'ervy'
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

  const totalChargesMax = data
    .map((v) => v.TotalCharges)
    .sort((a, b) => b - a)[0]
  const monthlyChargesMax = data
    .map((v) => v.MonthlyCharges)
    .sort((a, b) => b - a)[0]

  const features = data.map((v) => [
    v.TotalCharges / totalChargesMax,
    v.MonthlyCharges / monthlyChargesMax,
  ])
  const labels = data.map((v) => binary[v.Churn])

  const xs = tf.tensor2d(features, [features.length, 2])
  const ys = tf.tensor2d(labels, [labels.length, 1])

  // define model

  const model = tf.sequential()

  model.add(
    tf.layers.dense({ units: 1, inputShape: [2], activation: 'sigmoid' })
  )

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

  // [8000, 2000] => 8000 in total, an old customer. 2000 monthly, became very expensive. should predict a churn
  // [2000, 12] => 2000 in total, an average customer. 10 monthly, very cheap. low chances of a churn

  const predictData = [[2000, 10]].map((v) => [
    v[0] / totalChargesMax,
    v[1] / monthlyChargesMax,
  ])

  const predictTensor = tf.tensor2d(predictData, [predictData.length, 2])

  const predict = model.predict(predictTensor) as tf.Tensor
  const res = Array.from(predict.dataSync())
  console.log(res)
  console.log(res.map((v) => binary[Math.round(v)]))

  // console.log(
  //   ervy.bullet(
  //     predictData.map((v, k) => ({
  //       key: (v * totalChargesMax).toString(),
  //       value: res[k],
  //     }))
  //   )
  // )
}

main()
