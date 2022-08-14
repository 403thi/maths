pub struct LinearRegressionModel<'a> {
	x: &'a[f32],
	y: &'a[f32],
	pub a: f32,
	pub b: f32,
	pub summation_x: f32,
	pub summation_y: f32,
	pub summation_xy: f32,
	pub summation_x_square: f32,
	pub summation_y_square: f32,
	pub n: f32,
}

impl<'a> LinearRegressionModel<'a> {
	pub fn new(x: &'a[f32], y: &'a[f32]) -> Self {
		LinearRegressionModel {
			x: &x,
			y: &y,
			a: 0.0,
			b: 0.0,
			summation_x: 0.0,
			summation_y: 0.0,
			summation_xy: 0.0,
			summation_x_square: 0.0,
			summation_y_square: 0.0,
			n: 0.0,
		}
	}
	
	pub fn train(&mut self) {
		let xy = multiply_two_arrays(&self.x, &self.y);
		
		let x_square = multiply_two_arrays(&self.x, &self.x);
		let y_square = multiply_two_arrays(&self.y, &self.y);
		self.summation_x_square = sum_all(&x_square);
		self.summation_y_square  = sum_all(&y_square);
		
		self.summation_x  = sum_all(&self.x);
		self.summation_y  = sum_all(&self.y);
		self.summation_xy = sum_all(&xy);
		
		self.n = self.x.len() as f32;
		self.a = self.get_a();
		self.b = self.get_b();
	}
	
	pub fn predict(&self, x: f32) -> f32 {
		self.a + self.b * x
	}
	
	/*
		((Σy)(Σx^2) - (Σx)(Σxy)) / n(Σx^2)-(Σx)^2
		n = sample size
	*/
	pub fn get_a(&self) -> f32 {
		((self.summation_y * self.summation_x_square) - (self.summation_x * self.summation_xy)) /
		(self.n * self.summation_x_square - self.summation_x.powi(2))
	}
	
	/*
		(n(Σxy) - (Σx)(Σy)) / n(Σx^2) - (Σx)^2
	*/
	pub fn get_b(&self) -> f32 {
		((self.n * (self.summation_xy) - self.summation_x * self.summation_y)) /
		(self.n * self.summation_x_square - self.summation_x.powi(2))
	}
	
	/* Pearson correlation coefficient
		r = Sxy/sqrt(Sxx*Syy)
		Sxy = Σ(xᵢ*yᵢ) - ((Σxᵢ) * (Σyᵢ)) / n    x̄ ȳ
		Sxx = Σ(xᵢ^2) - (Σ(xᵢ^2))^2 / n
		Syy = Σ(yᵢ^2) - (Σ(yᵢ^2))^2 / n
		
	*/
	pub fn get_correlation_coefficient(&self) -> f32 {
		let sxy = self.summation_xy - (self.summation_x * self.summation_y) / self.n;
		let sxx = self.summation_x_square - self.summation_x.powi(2) / self.n;
		let syy = self.summation_y_square - self.summation_y.powi(2) / self.n;
		let r = sxy / (sxx*syy).sqrt();
		r
	}
}

fn sum_all(array: &[f32]) -> f32 {
	let mut result: f32 = 0.0;
	for element in array {
		result += element;
	}
	result
}

fn multiply_two_arrays(a: &[f32], b: &[f32]) -> Vec<f32> {
	
	let mut result: Vec<f32> = Vec::new();
	
	for (x,y) in a.iter().zip(b.iter()) {
		result.push(x*y);
	}
	
	result
}