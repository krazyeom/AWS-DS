//https://github.com/komiya-atsushi/xgboost-predictor-java

import biz.k11i.xgboost.Predictor
val predictor = Predictor(resources.assets.open("model.xgb"))

....

predict.setOnClickListener{

    //훈련용을 데이터를 받는다.
    val denseArray = doubleArrayOf(km.text.toString().toDouble(), age.text.toString().toDouble())

    //벡터로 변환
    val featureVector = FVec.Transformer.fromArray(denseArray, true);

    //예측
    val prediction = predictor.predict(featureVector)

    //실제 결과에 꽂는다.
    result_xgb.text = prediction[0].toString() + "만원";

    result_lr.text = (2128.2335 + -0.1706 * age.text.toString().toDouble() + -0.0052 * km.text.toString().toDouble()).toString() + "만원"

}
