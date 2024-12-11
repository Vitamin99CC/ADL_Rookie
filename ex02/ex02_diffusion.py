import torch
import torch.nn.functional as F
from ex02_helpers import extract
from tqdm import tqdm


def linear_beta_schedule(beta_start, beta_end, timesteps):
    """
    standard linear beta/variance schedule as proposed in the original paper
    """
    return torch.linspace(beta_start, beta_end, timesteps)


# TODO: Transform into task for students
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    # TODO (2.3): Implement cosine beta/variance schedule as discussed in the paper mentioned above
    t = torch.arange(0, timesteps, dtype=torch.float32)

    # Total timesteps (T - 1 for normalization)
    T = timesteps - 1

    # Compute alpha_bar_t (cosine schedule)
    alpha_bar_t = torch.cos(((t / T) + s) / (1 + s) * (torch.pi / 2)) ** 2

    # Compute beta_t using the formula
    beta_schedule = torch.zeros_like(alpha_bar_t)
    beta_schedule[0] = 1 - alpha_bar_t[0]  # Handle the first beta_t value
    beta_schedule[1:] = 1 - (alpha_bar_t[1:] / alpha_bar_t[:-1])  # Compute for t > 0
    return beta_schedule


def sigmoid_beta_schedule(beta_start, beta_end, timesteps):
    """
    sigmoidal beta schedule - following a sigmoid function
    """
    # TODO (2.3): Implement a sigmoidal beta schedule. Note: identify suitable limits of where you want to sample the sigmoid function.
    # Note that it saturates fairly fast for values -x << 0 << +x
        # Create an array of timesteps t ranging from 0 to T-1
    s_limit = 5
    t = torch.arange(0, timesteps)
    # Sigmoid function values
    sigmoid_values = 1 / (1 + np.exp(-(-s_limit + (2 * t / timesteps) * s_limit)))
    # Compute beta_t based on the formula
    beta_t = beta_start + sigmoid_values * (beta_end - beta_start)
    return beta_t


class Diffusion:

    # TODO (2.4): Adapt all methods in this class for the conditional case. You can use y=None to encode that you want to train the model fully unconditionally.

    def __init__(self, timesteps, get_noise_schedule, img_size, device="cuda"):
        """
        Takes the number of noising steps, a function for generating a noise schedule as well as the image size as input.
        """
        self.timesteps = timesteps

        self.img_size = img_size
        self.device = device

        # define beta schedule
        self.betas = get_noise_schedule(self.timesteps)

        # TODO (2.2): Compute the central values for the equation in the forward pass already here so you can quickly use them in the forward pass.
        # Note that the function torch.cumprod may be of help

        ### alpha_t = 1 - beta_t
        ### alphabar = cumuliative product of alpha_t
        # One instance for betas: linear scheduler
        # self.betas = torch.linspace(0.0001, 0.02, timesteps)
        # define alphas

        self.alphas = 1.0 - self.betas
        self.alphabar_t = torch.cumprod(self.alphas, dim=0)
        self.alphabar_t_minus_1 = F.pad(self.alphabar_t[:-1], (1, 0), value=1.)
        self.sqrt_alpha = torch.sqrt(self.alphas)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        # TODO
        ### = normal distribution (x_t; sqrt(1-beta_t)*x_{t-1}, beta_t*Identity)
        #mean = lambda x: torch.sqrt(1 - beta_t) * x
        #covariance = self.betas[] torch.eye(x_t_minus_1.size(-1), device=x_t_minus_1.device)
        #self.diffusion_q = lambda x: torch.distributions.MultivariateNormal(mean, covariance)
        self.sqrt_alphabar_t = torch.sqrt(1. - self.alphabar_t)
        self.sqrt_alphabar_tm1 = torch.sqrt(1. - self.alphabar_t_minus_1)
        self.sqrt_1_minus_alphabar_t = torch.sqrt(1. - self.alphabar_t)
        self.sqrt_1_minus_alphabar_t_minus_1 = torch.sqrt(1. - self.alphabar_t_minus_1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # TODO
        self.posterior_variance = self.betas * (1. - self.alphabar_t_minus_1) / (1. - self.alphabar_t)
        # self.mean = self.betas / torch.sqrt(1.0 - self.alphabar)
        # self.variance = torch.sqrt(self.betas)


    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        # TODO (2.2): implement the reverse diffusion process of the model for (noisy) samples x and timesteps t. Note that x and t both have a batch dimension
        b, *_ = x.shape
        batched_times = torch.full((b, ), t, device=self.device) # TODO: ???
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        preds = model(x, batched_times)

        
        noise = torch.randn_like(x, device=self.device) if t > 0 else 0
        variance = self.posterior_variance[t_index]
        x_t_minus_1 = 1./torch.sqrt(self.alphas[t_index]) * \
            (x - (1.-self.alphas[t_index])/(torch.sqrt(1-self.alphabar_t[t_index]))*preds) + \
            variance * noise
        # TODO (2.2): The method should return the image at timestep t-1.
        return x_t_minus_1
    
    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3):
        # TODO (2.2): Implement the full reverse diffusion loop from random noise to an image, iteratively ''reducing'' the noise in the generated image.
        img = torch.randn((batch_size,) + image_size, device=self.device)
        for t_index in tqdm (reversed(range(0, len(self.timesteps))), 
                       desc = 'sampling loop time step', total = self.num_timesteps):
            img = self.p_sample(model, img, self.timesteps[t_index], t_index)
        # TODO (2.2): Return the generated images
        return img
        
        

    # forward diffusion (using the nice property)
    def q_sample(self, x_zero, t, noise=None):

        # TODO (2.2): Implement the forward diffusion process using the beta-schedule defined in the constructor; 
        # if noise is None, you will need to create a new noise vector, otherwise use the provided one.
        if noise is None:
            noise = torch.randn_like(x_zero, device=self.device)
        
        x_t = extract(self.sqrt_alphabar_t, t, x_zero.shape) * x_zero + \
            extract(self.sqrt_1_minus_alphabar_t, t, x_zero.shape) * noise

        return x_t


    def p_losses(self, denoise_model, x_zero, t, noise=None, loss_type="l1"):
        # TODO (2.2): compute the input to the network using the forward diffusion process 
        # and predict the noise using the model; 
        # if noise is None, you will need to create a new noise vector, otherwise use the provided one.
        if noise is None:
            noise = torch.randn_like(x_zero, device=self.device)

        x = self.q_sample(x_zero, t, noise)
        pred = denoise_model(x, t)


        if loss_type == 'l1':
            # TODO (2.2): implement an L1 loss for this task
            loss = F.l1_loss(pred, x_zero, reduction="mean")
        elif loss_type == 'l2':
            # TODO (2.2): implement an L2 loss for this task
            loss = F.mse_loss(pred, x_zero, reduction="mean")
        else:
            raise NotImplementedError()

        return loss
