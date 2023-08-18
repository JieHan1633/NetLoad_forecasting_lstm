# measurements
import torch
import numpy as np

class measurements():
    def __init__(self, pred, target, regweight=5):
        self.pred = torch.Tensor(pred)
        self.target = torch.Tensor(target) 
        self.regweight = regweight
        
    def pinball_regloss_CityDayTime(self,daytimeIdx):
        # to be modified
        daytimeIdx = torch.Tensor(daytimeIdx)
        day_pred = self.pred(daytimeIdx==1)
        day_tar = self.target(daytimeIdx==1)
        # If self.target < self.pred, I = 1, else I = 0.   
        I = torch.where(day_tar[:, :, None]< day_pred, 1, 0) 
        # I = np.less(self.target[:, None], self.pred).astype(int)
        # L(τ, y, ŷ) = (ŷ - y)(I(y ≤ ŷ) - τ)  
        #              (ŷ - y) * -τ for y > ŷ
        #              (ŷ - y) * (1 - τ) for y ≤ ŷ  
        # L(τ, y, ŷ) = (y - ŷ)(τ - I(y ≤ ŷ))  
        #              (y - ŷ) * τ for y > ŷ
        #              (y - ŷ) * (τ - 1) for y ≤ ŷ 
        weights = torch.Tensor(np.array([0.1]*11 + [self.regweight])) 
        tau = torch.Tensor(np.reshape(np.arange(0,1.1,0.1),(1,11))) 
        pinball_mean =  torch.mean(tau * (1 - I) * ( day_tar[:, :,None]- day_pred) + \
                                   (tau - 1) * I * ( day_tar[:,:, None] -  day_pred))
        # pinball_loss =  np.mean(tau * (1 - I) * (self.target[:, None]-self.pred) + (tau - 1) * I * (self.target[:, None] - self.pred))
        
        # pinball_sum = np.sum(tau * (1 - I) * (self.target[:, None]-self.pred) + (tau - 1) * I * (self.target[:, None] - self.pred))
        
        # MSE for each percentile self.prediction
        mse_loss = torch.mean(( day_pred - day_tar[:, :, None])**2, axis=[0,1]) 
        # if self.pred.ndim == 2: 
        #     mse_loss = np.mean((self.pred - self.target[:, None])**2, axis=0)
        # elif pred.ndim == 3:
        #     mse_loss = np.mean((self.pred - self.target[:, None])**2, axis=(0,2))
        
        # Compute the weighted sum of all losses
        # weights in the same order as the quantiles for predictions. 
        # The last weight in the weights list is applied to the pinball loss.
        pinball_regloss = torch.dot(weights[0:-1], mse_loss) + weights[-1] * pinball_mean
        
        return pinball_regloss.item()
    
    def pinball_regloss(self): 
         
        I = torch.where(self.target[:, :, None]<self.pred, 1, 0) 
        tau = torch.Tensor(np.reshape(np.arange(0,1.1,0.1),(1,11)))
        # pinball_sum = torch.sum(tau * (1 - I) * (self.target[:, :,None]-self.pred) + (tau - 1) * I * (self.target[:, :,None] - self.pred))
        pinball_mean =  torch.mean(tau * (1 - I) * (self.target[:, :,None]-self.pred) + (tau - 1) * I * (self.target[:,:, None] - self.pred))
        
        weights = torch.Tensor(np.array([0.1]*11 + [self.regweight]))
         
        # mse should be [11]
        mse_loss = torch.mean((self.pred - self.target[:, :, None])**2, axis=[0,1]) 
        
        pinball_loss = torch.dot(weights[0:-1], mse_loss) + weights[-1] * pinball_mean 
        return pinball_loss.item()
    
    def pinball_loss(self):
        # The function can be used with either a single sample or multiple samples 
        # One sample:  
        # self.target has shape (n_output,), such as (24,)
        # self.pred has shape (n_output, n_quantiles), such as (24, 11)
        # tau has shape (1, n_quantiles,), such as (1, 11), and is a vector of quantiles
        # example: tau = np.linspace(0, 1, 11)[None, :] # Shape tau to match self.pred dimensions
        
        # Multiple samples:
        # self.target has shape (n_output, n_samples), such as (24, 100)
        # self.pred has shape (n_output, n_quantiles, n_samples), such as (24, 11, 100)
        # tau has shape (1, n_quantiles, 1), such as (1, 11, 1), and is a vector of quantiles
        # example: tau = np.linspace(0, 1, 11)[None, :, None] # Reshape tau to match self.pred dimensions
        
        # If self.target < self.pred, I = 1, else I = 0.   
        I = torch.where(self.target[:, :, None]<self.pred, 1, 0)
        # print("target shape:",self.target.shape)
        # I = np.less(self.target[:, None], self.pred).astype(int)
        # L(τ, y, ŷ) = (ŷ - y)(I(y ≤ ŷ) - τ)  
        #              (ŷ - y) * -τ for y > ŷ
        #              (ŷ - y) * (1 - τ) for y ≤ ŷ   
        # pinball_sum = np.sum(tau * (1 - I) * (self.target[:, None]-self.pred) + (tau - 1) * I * (self.target[:, None] - self.pred))
        # pinball_mean =  np.mean(tau * (1 - I) * (self.target[:, None]-self.pred) + (tau - 1) * I * (self.target[:, None] - self.pred))
        tau = torch.Tensor(np.reshape(np.arange(0,1.1,0.1),(1,11)))
        pinball_sum = torch.sum(tau * (1 - I) * (self.target[:, :,None]-self.pred) + (tau - 1) * I * (self.target[:, :,None] - self.pred))
        pinball_mean =  torch.mean(tau * (1 - I) * (self.target[:, :,None]-self.pred) + (tau - 1) * I * (self.target[:,:, None] - self.pred))
        
        return pinball_mean.item()
    
    def CRPS_dayloss(self,daytimeIdx):
        ### inp dtype: torch.Tensor
        ###     shape: (batch size, 24 hours' obs, 11 percentiles)
        ### target dtype: torch.Tensor
        ###     shape: (batch size, 24 hours' obs)
        ### inp and target should be on the same device 
        loss = [] 
        daytimeIdx = torch.Tensor(daytimeIdx)
        # day_pred = self.pred(daytimeIdx==1)
        # day_tar = self.target(daytimeIdx==1)
        if self.pred.dim()==3 and self.target.dim()==2:
            n = len(self.target)
            for i in range(n): 
                obs = self.target[i,:]
                fx = self.pred[i,:,:]
                idx = daytimeIdx[i,:] 
                obs = obs[idx==1] 
                hrs = len(obs)
                idx = idx.repeat(11,1)
                idx = torch.transpose(idx,1,0)
                fx = torch.reshape(fx[idx==1],(hrs,11))
                fx_prob = torch.Tensor(np.reshape(list(range(0,110,10))*hrs,(hrs,11)))
                tmp = continuous_ranked_probability_score(obs, fx, fx_prob) 
                loss.append(tmp) 
            loss = sum(loss)/len(loss)
            loss = loss.item()
        if self.pred.dim()==2 and self.target.dim()==1: 
            idx = daytimeIdx.flatten()
            obs = self.target[idx==1]
            fx = self.pred[idx==1]
            n = len(obs)
            fx_prob = torch.Tensor(np.reshape(list(range(0,110,10))*n,(n,11)))
            tmp = continuous_ranked_probability_score(obs, fx, fx_prob) 
            loss = tmp 
        # print("loss: ", loss)
        return loss
    
    def CRPS_array_dayloss(self, daytimeIdx):
        loss = [] 
        daytimeIdx = torch.Tensor(daytimeIdx) 
        assert (self.pred.dim()==3 and self.target.dim()==2) or (self.pred.dim()==2 and self.target.dim()==1), \
            'prediction should be n_days*24hours*n_percentiles and targets should be n_days*24hours'
        if self.pred.dim()==3 and self.target.dim()==2:
            n = len(self.target) 
            for i in range(n): 
                obs = self.target[i,:]
                fx = self.pred[i,:,:]
                idx = daytimeIdx[i,:] 
                obs = obs[idx==1] 
                hrs = len(obs)
                idx = idx.repeat(11,1)
                idx = torch.transpose(idx,1,0)
                fx = torch.reshape(fx[idx==1],(hrs,11))
                fx_prob = torch.Tensor(np.reshape(list(range(0,110,10))*hrs,(hrs,11)))
                tmp = continuous_ranked_probability_score(obs, fx, fx_prob) 
                loss.append(tmp.item()) 
        else: 
            if self.pred.dim()==2 and self.target.dim()==1: 
                n = len(self.target)
                idx = daytimeIdx.flatten() 
                for i in range(n):
                    if idx[i]==1: 
                        obs = torch.Tensor([self.target[i]])
                        fx = torch.reshape(self.pred[i,:], (1,-1))
                        # print(f"{i}th obs value: {obs}")
                        fx_prob = torch.Tensor(np.reshape(np.arange(0,110,10),(1,11)))
                        tmp = continuous_ranked_probability_score(obs, fx, fx_prob) 
                        loss.append(tmp.item()) 
                    else: 
                        continue 
        return loss
        
    def CRPS_loss(self): 
       
        ### inp dtype: torch.Tensor
        ###     shape: (batch size, 24 hours' obs, 11 percentiles)
        ### target dtype: torch.Tensor
        ###     shape: (batch size, 24 hours' obs)
        ### inp and target should be on the same device
        n = len(self.target)
        loss = []  
        # fx_prob = fx_prob
        if self.pred.dim()==3 and self.target.dim()==2:
            fx_prob = torch.Tensor(np.reshape(list(range(0,110,10))*self.pred.shape[1],(self.pred.shape[1],11)))
            for i in range(n): 
                obs = self.target[i,:]
                fx = self.pred[i,:,:]
                tmp = continuous_ranked_probability_score(obs, fx, fx_prob) 
                loss.append(tmp) 
            loss = sum(loss)/len(loss)
            loss=loss.item()
       
        if self.pred.dim()==2 and self.target.dim()==1:  
            fx_prob = torch.Tensor(np.reshape(list(range(0,110,10))*n,(n,11)))
            obs = self.target
            fx = self.pred
            tmp = continuous_ranked_probability_score(obs, fx, fx_prob) 
            loss = tmp
        # print("loss: ", loss)
        
        return loss
def continuous_ranked_probability_score(obs, fx, fx_prob):
    """Continuous Ranked Probability Score (CRPS).

    .. math::

        \\text{CRPS} = \\frac{1}{n} \\sum_{i=1}^n \\int_{-\\infty}^{\\infty}
        (F_i(x) - \\mathbf{1} \\{x \\geq y_i \\})^2 dx

    where :math:`F_i(x)` is the CDF of the forecast at time :math:`i`,
    :math:`y_i` is the observation at time :math:`i`, and :math:`\\mathbf{1}`
    is the indicator function that transforms the observation into a step
    function (1 if :math:`x \\geq y`, 0 if :math:`x < y`). In other words, the
    CRPS measures the difference between the forecast CDF and the empirical CDF
    of the observation. The CRPS has the same units as the observation. Lower
    CRPS values indicate more accurate forecasts, where a CRPS of 0 indicates a
    perfect forecast. [1]_ [2]_ [3]_

    Parameters
    ----------
    obs : (n,) array_like
        Observations (physical unit).
    fx : (n, d) array_like
        Forecasts (physical units) of the right-hand-side of a CDF with d
        intervals (d >= 2), e.g., fx = [10 MW, 20 MW, 30 MW] is interpreted as
        <= 10 MW, <= 20 MW, <= 30 MW.
    fx_prob : (n, d) array_like
        Probability [%] associated with the forecasts.

    Returns
    -------
    crps : float
        The Continuous Ranked Probability Score, with the same units as the
        observation.

    Raises
    ------
    ValueError
        If the forecasts have incorrect dimensions; either a) the forecasts are
        for a single sample (n=1) with d CDF intervals but are given as a 1D
        array with d values or b) the forecasts are given as 2D arrays (n,d)
        but do not contain at least 2 CDF intervals (i.e. d < 2).

    Notes
    -----
    The CRPS can be calculated analytically when the forecast CDF is of a
    continuous parametric distribution, e.g., Gaussian distribution. However,
    since the Solar Forecast Arbiter makes no assumptions regarding how a
    probabilistic forecast was generated, the CRPS is instead calculated using
    numerical integration of the discretized forecast CDF. Therefore, the
    accuracy of the CRPS calculation is limited by the precision of the
    forecast CDF. In practice, this means the forecast CDF should 1) consist of
    at least 10 intervals and 2) cover probabilities from 0% to 100%.

    References
    ----------
    .. [1] Matheson and Winkler (1976) "Scoring rules for continuous
           probability distributions." Management Science, vol. 22, pp.
           1087-1096. doi: 10.1287/mnsc.22.10.1087
    .. [2] Hersbach (2000) "Decomposition of the continuous ranked probability
           score for ensemble prediction systems." Weather Forecast, vol. 15,
           pp. 559-570. doi: 10.1175/1520-0434(2000)015<0559:DOTCRP>2.0.CO;2
    .. [3] Wilks (2019) "Statistical Methods in the Atmospheric Sciences", 4th
           ed. Oxford; Waltham, MA; Academic Press.

    """
    # match observations to fx shape: (n,) => (n, d)
    # if np.ndim(fx) < 2:
    if fx.dim()<2:
        raise ValueError("forecasts must be 2D arrays (expected (n,d), got"
                         f"{np.shape(fx)})")
    # elif np.shape(fx)[1] < 2:
    elif fx.shape[1] < 2:
        raise ValueError("forecasts must have d >= 2 CDF intervals "
                         f"(expected >= 2, got {np.shape(fx)[1]})")

    n = len(fx)

    # extend CDF min to ensure obs within forecast support
    # fx.shape = (n, d) ==> (n, d + 1)
    fx_min = torch.minimum(obs, fx[:, 0])
    fx = torch.hstack([fx_min[:, np.newaxis], fx])
    fx_prob = torch.hstack([torch.zeros([n, 1]), fx_prob])

    # extend CDF max to ensure obs within forecast support
    # fx.shape = (n, d + 1) ==> (n, d + 2)
    idx = (fx[:, -1] < obs)
    fx_max = torch.maximum(obs, fx[:, -1])
    # fx = torch.hstack([fx, fx_max[:, np.newaxis]])
    fx = torch.hstack([fx, fx_max[:, None]])
    fx_prob = torch.hstack([fx_prob, torch.full([n, 1], 100)])

    # indicator function:
    # - left of the obs is 0.0
    # - obs and right of the obs is 1.0
    # print(f"fx shape:{fx.shape}")
    # print(f"obs shape:{obs.shape}")
    o = torch.where(fx >= obs[:, None], 1.0, 0.0)

    # correct behavior when obs > max fx:
    # - should be 0 over range: max fx < x < obs
    o[idx, -1] = 0.0

    # forecast probabilities [unitless]
    f = fx_prob / 100.0

    # integrate along each sample, then average all samples
    crps = torch.mean(torch.trapezoid((f - o) ** 2, x=fx, axis=1))

    return crps

if __name__=="__main__":
    m = measurements(torch.randn([10,24,11]),torch.randn([10,24]),'Amity')
    a = m.pinball_regloss()
    b = m.CRPS_cityloss()
    c = m.pinball_regloss_CityDayTime()
    d = m.pinball_loss()
    print(a,b,c,d)