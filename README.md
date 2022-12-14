# MATHs - Repository for my math + rust studies

### [Simple Linear Regression Model](src/linear_regression)
**Use:**

```rs
use crate::linear_regression::model::LinearRegressionModel;
use std::time::Instant;
fn main() {
  let x: [f32; 20] = [
    20.0, 12.0, 14.0 ,15.0, 18.0, 9.0,
    5.0, 4.0, 8.0, 13.0, 14.0, 15.0,
    19.0, 18.0, 12.0, 11.0, 10.0, 15.0,
    17.0, 20.0
  ];
  let y: [f32; 20] = [
    9.5, 2.5, 3.6, 6.7, 5.2, 1.0,
    0.0, 1.5, 2.0, 3.0, 3.5, 4.5,
    8.5, 7.5, 5.0, 4.0, 3.0, 5.0,
    6.5, 10.0
  ];
  // create model with x and y inputs
  let mut model = LinearRegressionModel::new(&x, &y);
  // train model
  model.train();
  // predict
  let n = model.predict(16.0);
  println!("result: {n}");
  println!("coefficient a: {}", model.a);
  println!("coefficient b: {}", model.b);
  println!("correlation coefficient: {}", model.get_correlation_coefficient());
}
```