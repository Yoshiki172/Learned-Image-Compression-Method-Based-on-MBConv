import torch
import math
from proposal.bitEstimator import *
from proposal.Transform import *
from proposal.Hyper import *
from proposal.entropy import *


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.middle_channel_N = 256 #high=256,low=192
        self.out_channel_Y_M = 80 #low=48,high=64
        self.out_channel_Z_M = 192
        self.bitEstimator = BitEstimator(channel=self.out_channel_Z_M)
        self.Encoder = Analysis_transform(self.middle_channel_N,self.out_channel_Y_M)
        self.Decoder = Synthesis_transform(self.middle_channel_N,self.out_channel_Y_M)
        self.HyperEncoder = Analysis_Hyper(self.out_channel_Z_M,self.out_channel_Y_M)
        self.HyperDecoder = Synthesis_Hyper(self.out_channel_Z_M,self.out_channel_Y_M)
        self.entropy = Entropy(self.out_channel_Y_M)

    def feature_probs_based_GMM(self,feature, means, sigmas, weights):
            mean1 = means[:,:,:,:,0]
            mean2 = means[:,:,:,:,1]
            mean3 = means[:,:,:,:,2]
            sigma1 = sigmas[:,:,:,:,0]
            sigma2 = sigmas[:,:,:,:,1]
            sigma3 = sigmas[:,:,:,:,2]
            weight1 = weights[:,:,:,:,0]
            weight2 = weights[:,:,:,:,1]
            weight3 = weights[:,:,:,:,2]
    
            sigma1, sigma2, sigma3 = sigma1.clamp(1e-10, 1e10), sigma2.clamp(1e-10, 1e10), sigma3.clamp(1e-10, 1e10)
            
            gaussian1 = torch.distributions.laplace.Laplace(mean1, sigma1)
            gaussian2 = torch.distributions.laplace.Laplace(mean2, sigma2)
            gaussian3 = torch.distributions.laplace.Laplace(mean3, sigma3)
            prob1 = gaussian1.cdf(feature + 0.5) - gaussian1.cdf(feature - 0.5)
            prob2 = gaussian2.cdf(feature + 0.5) - gaussian2.cdf(feature - 0.5)
            prob3 = gaussian3.cdf(feature + 0.5) - gaussian3.cdf(feature - 0.5)
            
            
            probs = weight1 * prob1 + weight2 * prob2 + weight3 * prob3
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, probs
    def feature_probs_based_GLLMM(self,feature, means, sigmas, weights, probs_mix):
            mean1,mean2,mean3 = means[:,:,:,:,0],means[:,:,:,:,1],means[:,:,:,:,2]
            mean4,mean5,mean6 = means[:,:,:,:,3],means[:,:,:,:,4],means[:,:,:,:,5]
            mean7,mean8,mean9 = means[:,:,:,:,6],means[:,:,:,:,7],means[:,:,:,:,8]
            sigma1,sigma2,sigma3 = sigmas[:,:,:,:,0],sigmas[:,:,:,:,1],sigmas[:,:,:,:,2]
            sigma4,sigma5,sigma6 = sigmas[:,:,:,:,3],sigmas[:,:,:,:,4],sigmas[:,:,:,:,5]
            sigma7,sigma8,sigma9 = sigmas[:,:,:,:,6],sigmas[:,:,:,:,7],sigmas[:,:,:,:,8]
            weight1,weight2,weight3 = weights[:,:,:,:,0],weights[:,:,:,:,1],weights[:,:,:,:,2]
            weight4,weight5,weight6 = weights[:,:,:,:,3],weights[:,:,:,:,4],weights[:,:,:,:,5]
            weight7,weight8,weight9 = weights[:,:,:,:,6],weights[:,:,:,:,7],weights[:,:,:,:,8]
            prob_m0,prob_m1,prob_m2 = probs_mix[:,:,:,:,0],probs_mix[:,:,:,:,1],probs_mix[:,:,:,:,2]
    
            sigma1, sigma2, sigma3 = sigma1.clamp(1e-10, 1e10), sigma2.clamp(1e-10, 1e10), sigma3.clamp(1e-10, 1e10)
            sigma4, sigma5, sigma6 = sigma4.clamp(1e-10, 1e10), sigma5.clamp(1e-10, 1e10), sigma6.clamp(1e-10, 1e10)
            sigma7, sigma8, sigma9 = sigma7.clamp(1e-10, 1e10), sigma8.clamp(1e-10, 1e10), sigma9.clamp(1e-10, 1e10)
            gaussian1 = torch.distributions.normal.Normal(mean1, sigma1)
            gaussian2 = torch.distributions.normal.Normal(mean2, sigma2)
            gaussian3 = torch.distributions.normal.Normal(mean3, sigma3)
            laplace1 = torch.distributions.laplace.Laplace(mean4, sigma4)
            laplace2 = torch.distributions.laplace.Laplace(mean5, sigma5)
            laplace3 = torch.distributions.laplace.Laplace(mean6, sigma6)
            
            UNI = torch.distributions.Uniform(0, 1)
            print(UNI)
            logit1 = torch.distributions.TransformedDistribution(UNI, [torch.distributions.SigmoidTransform().inv, torch.distributions.AffineTransform(mean7, sigma7)])
            #logit1 = torch.distributions.TransformedDistribution(torch.distributions.Uniform(0, 1), [torch.distributions.SigmoidTransform().inv, torch.distributions.AffineTransform(mean7, sigma7)])
            logit2 = torch.distributions.TransformedDistribution(torch.distributions.Uniform(0, 1), [torch.distributions.SigmoidTransform().inv, torch.distributions.AffineTransform(mean8, sigma8)])
            logit3 = torch.distributions.TransformedDistribution(torch.distributions.Uniform(0, 1), [torch.distributions.SigmoidTransform().inv, torch.distributions.AffineTransform(mean9, sigma9)])
            
            prob1 = gaussian1.cdf(feature + 0.5) - gaussian1.cdf(feature - 0.5)
            prob2 = gaussian2.cdf(feature + 0.5) - gaussian2.cdf(feature - 0.5)
            prob3 = gaussian3.cdf(feature + 0.5) - gaussian3.cdf(feature - 0.5)
            prob4 = laplace1.cdf(feature + 0.5) - laplace1.cdf(feature - 0.5)
            prob5 = laplace2.cdf(feature + 0.5) - laplace2.cdf(feature - 0.5)
            prob6 = laplace3.cdf(feature + 0.5) - laplace3.cdf(feature - 0.5)
            prob7 = logit1.cdf(feature + 0.5) - logit1.cdf(feature - 0.5)
            prob8 = logit2.cdf(feature + 0.5) - logit2.cdf(feature - 0.5)
            prob9 = logit3.cdf(feature + 0.5) - logit3.cdf(feature - 0.5)

            probs = prob_m0*(weight1 * prob1 + weight2 * prob2 + weight3 * prob3) + \
                prob_m1*(weight4 * prob4 + weight5 * prob5 + weight6 * prob6) + \
                prob_m2*(weight7 * prob7 + weight8 * prob8 + weight9 * prob9)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, probs

    def feature_probs_based_sigma(self,feature, sigma):
            mu = torch.zeros_like(sigma)
            sigma = sigma.clamp(1e-10, 1e10)
            sigma = torch.nan_to_num(sigma, nan=1e-10)
            gaussian = torch.distributions.laplace.Laplace(mu, sigma)
            feature = torch.nan_to_num(feature,nan=1e-10)
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, probs
    
    def iclr18_estimate_bits_z(self,z):
            prob = self.bitEstimator(z + 0.5) - self.bitEstimator(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, prob
    
    def forward(self, inputs):    
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        quant_noise_feature = torch.zeros(inputs.size(0), self.out_channel_Y_M, inputs.size(2) // 8,inputs.size(3) // 8).to(device)
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
        
        quant_noise_z = torch.zeros(inputs.size(0), self.out_channel_Z_M, inputs.size(2) // 64, inputs.size(3) // 64).to(device)
        quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(quant_noise_z), -0.5, 0.5)

        Y = self.Encoder(inputs)
        batch_size = inputs.size()[0]

        LatentZ = self.HyperEncoder(Y)
        
        if self.training:
            compressed_Z = LatentZ + quant_noise_z
        else:
            compressed_Z = torch.round(LatentZ)
        
        phi = self.HyperDecoder(compressed_Z)
        
        if self.training:
            compressed_Y = Y + quant_noise_feature
        else:
            compressed_Y = torch.round(Y)
        
        means, sigmas, weights = self.entropy(phi,compressed_Y)
        #means, sigmas, weights,weights_mix = self.entropy(phi,compressed_Y)
     
        outputs = self.Decoder(compressed_Y)
        
        Y_bits, _ = self.feature_probs_based_GMM(compressed_Y, means, sigmas, weights)
        #Y_bits, _ = self.feature_probs_based_GLLMM(compressed_Y, means, sigmas, weights, weights_mix)
        
        Z_bits, _ = self.iclr18_estimate_bits_z(compressed_Z)
  
        mse_loss = torch.mean((outputs - inputs).pow(2))
        total_z_bpp = (Z_bits)/(batch_size*inputs.shape[2]*inputs.shape[3])
        total_y_bpp = (Y_bits)/(batch_size*inputs.shape[2]*inputs.shape[3])
        total_bpp = total_y_bpp + total_z_bpp
        
        #edge_loss
        """
        avg_kernel = torch.FloatTensor([[1/9, 1/9, 1/9], 
                                    [1/9, 1/9, 1/9], 
                                    [1/9, 1/9, 1/9]])
        avg_filter = avg_kernel.expand(3, 3, 3, 3).to(device)
        avg_inputs = F.conv2d(inputs, avg_filter,padding=1)
        avg_outputs = F.conv2d(outputs, avg_filter,padding=1)
        
        lap_kernel = torch.FloatTensor([[1, 1, 1], 
                                        [1, -8, 1], 
                                        [1, 1, 1]])
        lap_filter = lap_kernel.expand(3, 3, 3, 3).to(device)

        lap_inputs = F.conv2d(inputs, lap_filter,padding=1)
        lap_recon = F.conv2d(outputs, lap_filter,padding=1)
        lap_mse_loss = torch.mean((lap_recon - lap_inputs).pow(2))
        """
        return outputs,mse_loss,total_bpp,total_y_bpp,total_z_bpp
    