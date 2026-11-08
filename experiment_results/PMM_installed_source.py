class PMM(_BaseLoss):
    r"""Poisson Mixture Mesh

    This Poisson Mixture statistical model assumes independence across groups of
    data $\mathcal{G}=\{[g_{i}]\}$, and estimates relationships within the group.

    ```math
    \mathrm{P}\left(\mathbf{y}_{[b][t+1:t+H]}\right) =
    \prod_{ [g_{i}] \in \mathcal{G}} \mathrm{P} \left(\mathbf{y}_{[g_{i}][\tau]} \right) =
    \prod_{\beta\in[g_{i}]}
    \left(\sum_{k=1}^{K} w_k \prod_{(\beta,\tau) \in [g_i][t+1:t+H]} \mathrm{Poisson}(y_{\beta,\tau}, \hat{\lambda}_{\beta,\tau,k}) \right)
    ```

    Args:
        n_components (int, optional): The number of mixture components. Defaults to 10.
        level (float list, optional): Confidence levels for prediction intervals. Defaults to [80, 90].
        quantiles (float list, optional): Alternative to level list, target quantiles. Defaults to None.
        return_params (bool, optional): Whether or not return the Distribution parameters. Defaults to False.
        batch_correlation (bool, optional): Whether or not model batch correlations. Defaults to False.
        horizon_correlation (bool, optional): Whether or not model horizon correlations. Defaults to False.

    References:
        - [Kin G. Olivares, O. Nganba Meetei, Ruijun Ma, Rohan Reddy, Mengfei Cao, Lee Dicker. Probabilistic Hierarchical Forecasting with Deep Poisson Mixtures. Submitted to the International Journal Forecasting, Working paper available at arxiv.](https://arxiv.org/pdf/2110.13179.pdf)
    """

    def __init__(
        self,
        n_components=10,
        level=[80, 90],
        quantiles=None,
        num_samples=1000,
        return_params=False,
        batch_correlation=False,
        horizon_correlation=False,
        weighted=False,
    ):
        super(PMM, self).__init__()
        # Transform level to MQLoss parameters
        qs, self.output_names = level_to_outputs(level)
        qs = torch.Tensor(qs)

        # Transform quantiles to homogeneous output names
        if quantiles is not None:
            quantiles, self.output_names = quantiles_to_outputs(quantiles)
            qs = torch.Tensor(quantiles)
        self.quantiles = torch.nn.Parameter(qs, requires_grad=False)
        self.num_samples = num_samples
        self.batch_correlation = batch_correlation
        self.horizon_correlation = horizon_correlation
        self.weighted = weighted

        # If True, predict_step will return Distribution's parameters
        self.return_params = return_params

        lambda_names = [f"-lambda-{i}" for i in range(1, n_components + 1)]
        if weighted:
            weight_names = [f"-weight-{i}" for i in range(1, n_components + 1)]
            self.param_names = [i for j in zip(lambda_names, weight_names) for i in j]
        else:
            self.param_names = lambda_names

        if self.return_params:
            self.output_names = self.output_names + self.param_names

        # Add first output entry for the sample_mean
        self.output_names.insert(0, "")

        self.n_outputs = 1 + weighted
        self.n_components = n_components
        self.outputsize_multiplier = self.n_outputs * n_components
        self.is_distribution_output = True
        self.has_predicted = False

    def domain_map(self, output: torch.Tensor):
        output = output.reshape(
            output.shape[0], output.shape[1], -1, self.outputsize_multiplier
        )

        return torch.tensor_split(output, self.n_outputs, dim=-1)

    def scale_decouple(
        self,
        output,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ):
        """Scale Decouple

        Stabilizes model's output optimization, by learning residual
        variance and residual location based on anchoring `loc`, `scale`.
        Also adds domain protection to the distribution parameters.
        """
        if self.weighted:
            lambdas, weights = output
            weights = F.softmax(weights, dim=-1)
        else:
            lambdas = output[0]

        if (loc is not None) and (scale is not None):
            if loc.ndim == 3:
                loc = loc.unsqueeze(-1)
                scale = scale.unsqueeze(-1)
            lambdas = (lambdas * scale) + loc

        lambdas = F.softplus(lambdas) + 1e-3

        if self.weighted:
            return (lambdas, weights)
        else:
            return (lambdas,)

    def get_distribution(self, distr_args) -> Distribution:
        """
        Construct the associated Pytorch Distribution, given the collection of
        constructor arguments and, optionally, location and scale tensors.

        Args:
            distr_args (torch.Tensor): Constructor arguments for the underlying Distribution type.

        Returns:
            Distribution: AffineTransformed distribution.
        """
        if self.weighted:
            lambdas, weights = distr_args
        else:
            lambdas = distr_args[0]
            weights = torch.full_like(lambdas, fill_value=1 / self.n_components)

        mix = Categorical(weights)
        components = Poisson(rate=lambdas)
        components.support = constraints.nonnegative
        distr = MixtureSameFamily(
            mixture_distribution=mix, component_distribution=components
        )

        self.distr_mean = distr.mean

        return distr

    def sample(self, distr_args: torch.Tensor, num_samples: Optional[int] = None):
        """
        Construct the empirical quantiles from the estimated Distribution,
        sampling from it `num_samples` independently.

        Args:
            distr_args (torch.Tensor): Constructor arguments for the underlying Distribution type.
            num_samples (int, optional): Overwrite number of samples for the empirical quantiles. Defaults to None.

        Returns:
            tuple: Tuple with samples, sample mean, and quantiles.
        """
        if num_samples is None:
            num_samples = self.num_samples

        # Instantiate Scaled Decoupled Distribution
        distr = self.get_distribution(distr_args=distr_args)
        samples = distr.sample(sample_shape=(num_samples,))
        samples = samples.permute(
            1, 2, 3, 0
        )  # [samples, B, H, N] -> [B, H, N, samples]

        sample_mean = torch.mean(samples, dim=-1, keepdim=True)

        # Compute quantiles
        quantiles_device = self.quantiles.to(distr_args[0].device)
        quants = torch.quantile(input=samples, q=quantiles_device, dim=-1)
        quants = quants.permute(1, 2, 3, 0)  # [Q, B, H, N] -> [B, H, N, Q]

        return samples, sample_mean, quants

    def update_quantile(self, q: Optional[List[float]] = None):
        if q is not None:
            self.quantiles = nn.Parameter(
                torch.tensor(q, dtype=torch.float32), requires_grad=False
            )
            self.output_names = (
                [""]
                + [f"_ql{q_i}" for q_i in q]
                + self.return_params * self.param_names
            )
            self.has_predicted = True
        elif q is None and self.has_predicted:
            self.quantiles = nn.Parameter(
                torch.tensor([0.5], dtype=torch.float32), requires_grad=False
            )
            self.output_names = ["", "-median"] + self.return_params * self.param_names

    def __call__(
        self,
        y: torch.Tensor,
        distr_args: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
    ):
        r"""
        Computes the negative log-likelihood objective function.
        To estimate the following predictive distribution:

        ```math
        \mathrm{P}(\mathbf{y}_{\tau}\,|\,\theta) \quad \mathrm{and} \quad -\log(\mathrm{P}(\mathbf{y}_{\tau}\,|\,\theta))
        ```

        where $\theta$ represents the distributions parameters. It aditionally
        summarizes the objective signal using a weighted average using the `mask` tensor.

        Args:
            y (torch.Tensor): Actual values.
            distr_args (torch.Tensor): Constructor arguments for the underlying Distribution type.
            mask (Union[torch.Tensor, None], optional): Specifies date stamps per serie to consider in loss. Defaults to None.

        Returns:
            float: Weighted loss function against which backpropagation will be performed.
        """
        # Instantiate Scaled Decoupled Distribution
        distr = self.get_distribution(distr_args=distr_args)
        x = distr._pad(y)
        log_prob_x = distr.component_distribution.log_prob(x)
        log_mix_prob = torch.log_softmax(distr.mixture_distribution.logits, dim=-1)
        if self.batch_correlation:
            log_prob_x = torch.sum(log_prob_x, dim=0, keepdim=True)
        if self.horizon_correlation:
            log_prob_x = torch.sum(log_prob_x, dim=1, keepdim=True)

        loss_values = -torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)

        return weighted_average(loss_values, weights=mask)
