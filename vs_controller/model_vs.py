import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device('cuda:0')
class Perception(nn.Module):
	def __init__(self, input_channels, h=256, w=256):
		super(Perception, self).__init__()
		
		self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(64)
		self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
		self.bn4 = nn.BatchNorm2d(64)
		self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
		self.bn5 = nn.BatchNorm2d(64)
		self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

	def forward(self, state):
		"""
		Params state and actions are torch tensors
		"""
		## convert state from a 2d image to a vector
		state = F.relu(self.bn1(self.conv1(state)))
		state = self.pool(state)
		state = F.relu(self.bn2(self.conv2(state)))
		state = self.pool(state)
		state = F.relu(self.bn3(self.conv3(state)))
		state = self.pool(state)
		#print('state.shape = {}'.format(state.shape))
		state = F.relu(self.bn4(self.conv4(state)))
		state = self.pool(state)
		#print('state.shape = {}'.format(state.shape))
		state = F.relu(self.bn5(self.conv5(state)))
		state = self.pool(state)
		#print('state.shape = {}'.format(state.shape))
		## add feature normalization
		state = featureL2Norm(state)
		#print('state.shape = {}'.format(state.shape))
		return state

class Perception_multiscale(nn.Module):
	def __init__(self, input_channels, h=256, w=256):
		super(Perception_multiscale, self).__init__()
		
		self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(64)
		self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
		self.bn4 = nn.BatchNorm2d(64)
		self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
		self.bn5 = nn.BatchNorm2d(64)
		self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		self.pool2 = torch.nn.MaxPool2d(kernel_size=4, stride=4, padding=0)

	def forward(self, state):
		"""
		Params state and actions are torch tensors
		"""
		## convert state from a 2d image to a vector
		state1 = F.relu(self.bn1(self.conv1(state)))
		state1 = self.pool(state1) ## batch x 16 x 64 x 64
		
		state2 = F.relu(self.bn2(self.conv2(state1)))
		state2 = self.pool(state2) ## batch x 32 x 32 x 32
		
		state3 = F.relu(self.bn3(self.conv3(state2)))
		state3 = self.pool(state3) ## batch x 64 x 16 x 16
		
		state4 = F.relu(self.bn4(self.conv4(state3)))
		state4 = self.pool(state4) ## batch x 64 x 8 x 8
		
		state5 = F.relu(self.bn5(self.conv5(state4)))
		state5 = self.pool(state5) ## batch x 64 x 4 x 4

		state4 = self.pool(state4) ## batch x 64 x 4 x 4
		#print('state4.shape = {}'.format(state4.shape))
		state3 = self.pool2(state3)## batch x 64 x 4 x 4
		#print('state3.shape = {}'.format(state3.shape))
		state = torch.cat((state5, state4, state3), dim=1) ## batch x 192 x 4 x 4
		#print('state.shape = {}'.format(state.shape))
		state = featureL2Norm(state)

		return state



def featureL2Norm(feature):
	epsilon = 1e-6
	norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature) ## batch x 16 x 4 x 4
	#print('norm.shape = {}'.format(norm.shape))
	#print('norm = {}'.format(norm))
	return torch.div(feature, norm)

class FeatureCorrelation(torch.nn.Module):
	def __init__(self):
		super(FeatureCorrelation, self).__init__()

	def forward(self, feature_A, feature_B):
		b,c,h,w = feature_A.size() # batch x 64 x 4 x 4
		# reshape features for matrix multiplication
		feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h*w) ## batch x 64 x 16
		feature_B = feature_B.view(b, c, h*w).transpose(1, 2) ## batch x 16 x 64
		# perform matrix mult.
		feature_mul = torch.bmm(feature_B, feature_A) ## batch x 16 x 16
		#print('feature_mul.shape = {}'.format(feature_mul.shape))
		correlation_tensor = feature_mul.view(b, h, w, h*w).transpose(2,3).transpose(1,2) ## batch x 16 x 4 x 4
		#print('correlation_tensor.shape = {}'.format(correlation_tensor.shape))       
		correlation_tensor = featureL2Norm(F.relu(correlation_tensor)) ## batch x 16 x 4 x 4
		#print('sum(correlation_tensor) = {}'.format(torch.sum(correlation_tensor[0, :, 0, 0])))
		return correlation_tensor

## compute correlation map between left_img and right_img
class VS_correlation(nn.Module):
	def __init__(self, perception_module, input_size=256):
		super(VS_correlation, self).__init__()
		self.perception = perception_module
		self.FeatureCorrelation = FeatureCorrelation()

	def forward(self, left_img, right_img):
		left_state = self.perception(left_img)
		#print('left_state.shape = {}'.format(left_state.shape))
		right_state = self.perception(right_img)
		corr = self.FeatureCorrelation(left_state, right_state) # batch x 16 x 4 x 4
		#corr = corr.reshape(corr.size(0), -1)
		return corr

class Forward_Dynamics(nn.Module):
	def __init__(self, correlation_module, input_size=257, output_size=256):
		super(Forward_Dynamics, self).__init__()
		self.correlation = correlation_module
		#self.linear = nn.Linear(input_size, output_size)
		self.conv = nn.Conv2d(17, 16, kernel_size=1, stride=1)
		self.bn = nn.BatchNorm2d(16)

	def forward(self, left_img, right_img, action):
		corr = self.correlation(left_img, right_img)
		## concatenate corr with action map
		#'''
		batch_size, _, b, c = corr.shape
		action_tensor = torch.ones((batch_size, 1, b, c)).to(device) # batch x 16 x 4 x 4
		for i in range(batch_size):
			action_tensor[i] = action_tensor[i] * action[i]
		#'''
		#print('corr.shape = {}'.format(corr.shape))
		#print('action = {}'.format(action))
		
		#print('state.shape = {}'.format(state.shape))
		state = torch.cat((corr, action_tensor), dim=1) ## batch x 17 x 4 x 4
		#print('state.shape = {}'.format(state.shape))
		x = F.relu(self.bn(self.conv(state)))
		#print('x.shape = {}'.format(x.shape))
		'''
		state = torch.cat((corr, action), dim=1) ## batch x 257
		x = self.linear(state)
		'''
		return x

class Inverse_Dynamics(nn.Module):
	def __init__(self, correlation_module, input_size=256, output_size=7):
		super(Inverse_Dynamics, self).__init__()
		self.correlation = correlation_module
		#self.conv = nn.Conv2d(32, 16, kernel_size=1, stride=1)
		#self.bn = nn.BatchNorm2d(16)
		self.action_fc = nn.Linear(input_size, output_size)

	def forward(self, left_img, next_img, goal_img):
		'''
		corr_left = self.correlation(left_img, goal_img) # batch x 256
		corr_next = self.correlation(next_img, goal_img) # batch x 256

		state = torch.cat((corr_left, corr_next), dim=1) ## batch x 512
		print('state.shape = {}'.format(state.shape))
		state = F.relu(self.bn(self.conv(state)))
		state = state.reshape(state.size(0), -1)
		print('state.shape = {}'.format(state.shape))
		'''
		#corr = self.correlation(left_img, next_img)
		corr = self.correlation(next_img, left_img)
		state = corr.reshape(corr.size(0), -1)

		action_prediction = self.action_fc(state)
		#print('action_prediction.shape = {}'.format(action_prediction.shape))
		output_action = F.softmax(action_prediction, dim=1)
		return output_action

class Inverse_Dynamics_error_state(nn.Module):
	def __init__(self, correlation_module, input_size=256, output_size=7):
		super(Inverse_Dynamics_error_state, self).__init__()
		self.correlation = correlation_module
		self.conv = nn.Conv2d(32, 16, kernel_size=1, stride=1)
		self.bn = nn.BatchNorm2d(16)
		self.action_fc = nn.Linear(input_size, output_size)

	def forward(self, left_img, next_img, goal_img):
		
		corr_left = self.correlation(goal_img, left_img) # batch x 16 x 4 x 4
		corr_next = self.correlation(goal_img, next_img) # batch x 16 x 4 x 4

		state = torch.cat((corr_left, corr_next), dim=1) ## batch x 32 x 4 x 4
		#print('state.shape = {}'.format(state.shape))
		state = F.relu(self.bn(self.conv(state))) ## batch x 16 x 4 x 4
		#print('state.shape = {}'.format(state.shape))
		state = state.reshape(state.size(0), -1)
		#print('state.shape = {}'.format(state.shape))

		action_prediction = self.action_fc(state)
		#print('action_prediction.shape = {}'.format(action_prediction.shape))
		output_action = F.softmax(action_prediction, dim=1)
		return output_action

class DQN_VS_Controller(nn.Module):
	## Actor's input_size is different from Critic's
	def __init__(self, correlation_module, output_size, input_size=256):
		super(DQN_VS_Controller, self).__init__()
		self.correlation = correlation_module

		self.linear = nn.Linear(input_size, output_size)

	def forward(self, left_img, goal_img):
		corr = self.correlation(goal_img, left_img)
		state = corr.reshape(corr.size(0), -1)

		x = self.linear(state)
		return x

class Perception_overlap(nn.Module):
	def __init__(self, input_channels, h=256, w=256):
		super(Perception_overlap, self).__init__()
		self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(32)
		'''
		self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(32)
		self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		## fully convolutional layer
		self.conv4 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
		'''
		self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(64)
		self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		## fully convolutional layer
		self.conv4 = nn.Conv2d(64, 1, kernel_size=1, stride=1)
		self.bn4 = nn.BatchNorm2d(1)

	def forward(self, state):
		state = F.relu(self.bn1(self.conv1(state)))
		state = self.pool(state)
		state = F.relu(self.bn2(self.conv2(state)))
		state = self.pool(state)
		state = F.relu(self.bn3(self.conv3(state)))
		state = self.pool(state) ## num_steps x 32 x 3 x 3
		state = F.relu(self.bn4(self.conv4(state)))
		state = state.reshape(state.size(0), -1)
		return state

class Preception_overlap_resnet(nn.Module):
	def __init__(self, input_channels, h=256, w=256):
		super(Preception_overlap_resnet, self).__init__()
		self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(64)
		self.conv4 = nn.Conv2d(64, 1, kernel_size=1, stride=1)
		self.bn4 = nn.BatchNorm2d(1)
		self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
	
	def forward(self, state):
		assert state.shape[1]==4
		state1 = state[:, :2, :, :]
		state2 = state[:, 2:4, :, :]

		state1 = F.relu(self.bn1(self.conv1(state1)))
		state1 = self.pool(state1)
		state1 = F.relu(self.bn2(self.conv2(state1)))
		state1 = self.pool(state1)
		state1 = F.relu(self.bn3(self.conv3(state1)))
		state1 = self.pool(state1) 
		state1 = F.relu(self.bn4(self.conv4(state1)))
		state1 = state1.reshape(state1.size(0), -1)

		state2 = F.relu(self.bn1(self.conv1(state2)))
		state2 = self.pool(state2)
		state2 = F.relu(self.bn2(self.conv2(state2)))
		state2 = self.pool(state2)
		state2 = F.relu(self.bn3(self.conv3(state2)))
		state2 = self.pool(state2) 
		state2 = F.relu(self.bn4(self.conv4(state2)))
		state2 = state2.reshape(state2.size(0), -1)

		x = torch.cat((state1, state2), 1)
		#print('x.shape = {}'.format(x.shape))
		return x

class DQN_OVERLAP_RESNET_Controller(nn.Module):
	## Actor's input_size is different from Critic's
	def __init__(self, perception_module, output_size, input_size=256, input_channels=2):
		super(DQN_OVERLAP_RESNET_Controller, self).__init__()
		self.perception = perception_module
		self.intermediate_fc = nn.Linear(input_size, 256)
		self.linear = nn.Linear(256, output_size)

	def forward(self, state):
		state = self.perception(state)
		x = self.linear(F.relu(self.intermediate_fc(state)))
		return x


class Perception_siamese(nn.Module):
	def __init__(self, input_channels=4, h=256, w=256):
		super(Perception_siamese, self).__init__()
		self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(64)
		self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		## fully convolutional layer
		self.conv4 = nn.Conv2d(64, 1, kernel_size=1, stride=1)
		self.bn4 = nn.BatchNorm2d(1)

		self.conv1_2 = nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2)
		self.bn1_2 = nn.BatchNorm2d(16)
		self.conv2_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
		self.bn2_2 = nn.BatchNorm2d(32)
		self.conv3_2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
		self.bn3_2 = nn.BatchNorm2d(64)
		self.pool_2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		## fully convolutional layer
		self.conv4_2 = nn.Conv2d(64, 1, kernel_size=1, stride=1)
		self.bn4_2 = nn.BatchNorm2d(1)

	def forward(self, state):
		assert state.shape[1]==4
		state1 = state[:, :2, :, :]
		state2 = state[:, 2:4, :, :]

		state1 = F.relu(self.bn1(self.conv1(state1)))
		state1 = self.pool(state1)
		state1 = F.relu(self.bn2(self.conv2(state1)))
		state1 = self.pool(state1)
		state1 = F.relu(self.bn3(self.conv3(state1)))
		state1 = self.pool(state1) ## num_steps x 32 x 3 x 3
		state1 = F.relu(self.bn4(self.conv4(state1)))
		state1 = state1.reshape(state1.size(0), -1)

		state2 = F.relu(self.bn1_2(self.conv1_2(state2)))
		state2 = self.pool(state2)
		state2 = F.relu(self.bn2_2(self.conv2_2(state2)))
		state2 = self.pool(state2)
		state2 = F.relu(self.bn3_2(self.conv3_2(state2)))
		state2 = self.pool(state2) ## num_steps x 32 x 3 x 3
		state2 = F.relu(self.bn4_2(self.conv4_2(state2)))
		state2 = state2.reshape(state2.size(0), -1)

		state = torch.cat((state1, state2), 1)
		#print('state.shape = {}'.format(state.shape))
		return state

class Perception_siamese_fusion(nn.Module):
	def __init__(self, input_channels=4, h=256, w=256):
		super(Perception_siamese_fusion, self).__init__()
		self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(64)
		self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		## fully convolutional layer
		self.conv4 = nn.Conv2d(64, 1, kernel_size=1, stride=1)
		self.bn4 = nn.BatchNorm2d(1)

		self.conv1_2 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)
		self.bn1_2 = nn.BatchNorm2d(16)
		self.conv2_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
		self.bn2_2 = nn.BatchNorm2d(32)
		self.conv3_2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
		self.bn3_2 = nn.BatchNorm2d(64)
		self.pool_2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		## fully convolutional layer
		self.conv4_2 = nn.Conv2d(64, 1, kernel_size=1, stride=1)
		self.bn4_2 = nn.BatchNorm2d(1)

		self.conv5 = nn.Conv2d(2, 1, kernel_size=1, stride=1)
		self.bn5 = nn.BatchNorm2d(1)

	def forward(self, state):
		assert state.shape[1]==3
		state1 = state[:, :2, :, :]
		state2 = state[:, 2, :, :].unsqueeze(1)
		#print('state2.shape = {}'.format(state2.shape))

		state1 = F.relu(self.bn1(self.conv1(state1)))
		state1 = self.pool(state1)
		state1 = F.relu(self.bn2(self.conv2(state1)))
		state1 = self.pool(state1)
		state1 = F.relu(self.bn3(self.conv3(state1)))
		state1 = self.pool(state1) ## num_steps x 32 x 3 x 3
		state1 = F.relu(self.bn4(self.conv4(state1)))

		state2 = F.relu(self.bn1_2(self.conv1_2(state2)))
		state2 = self.pool(state2)
		state2 = F.relu(self.bn2_2(self.conv2_2(state2)))
		state2 = self.pool(state2)
		state2 = F.relu(self.bn3_2(self.conv3_2(state2)))
		state2 = self.pool(state2) ## num_steps x 32 x 3 x 3
		state2 = F.relu(self.bn4_2(self.conv4_2(state2)))

		state = torch.cat((state1, state2), 1)
		#print('state.shape = {}'.format(state.shape))
		state = F.relu(self.bn5(self.conv5(state)))
		#print('state.shape = {}'.format(state.shape))

		state = state.reshape(state.size(0), -1)
		#print('state.shape = {}'.format(state.shape))
		return state

class Perception_siamese_fusion_old(nn.Module):
	def __init__(self, input_channels=4, h=256, w=256):
		super(Perception_siamese_fusion_new, self).__init__()
		self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(64)
		self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		## fully convolutional layer
		#self.conv4 = nn.Conv2d(64, 1, kernel_size=1, stride=1)
		#self.bn4 = nn.BatchNorm2d(1)

		self.conv1_2 = nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2)
		self.bn1_2 = nn.BatchNorm2d(8)
		self.conv2_2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
		self.bn2_2 = nn.BatchNorm2d(16)
		self.conv3_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
		self.bn3_2 = nn.BatchNorm2d(32)
		self.pool_2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		## fully convolutional layer
		#self.conv4_2 = nn.Conv2d(64, 1, kernel_size=1, stride=1)
		#self.bn4_2 = nn.BatchNorm2d(1)

		self.conv5 = nn.Conv2d(96, 1, kernel_size=1, stride=1)
		self.bn5 = nn.BatchNorm2d(1)

	def forward(self, state):
		assert state.shape[1]==3
		state1 = state[:, :2, :, :]
		state2 = state[:, 2, :, :].unsqueeze(1)
		#print('state2.shape = {}'.format(state2.shape))

		state1 = F.relu(self.bn1(self.conv1(state1)))
		state1 = self.pool(state1)
		state1 = F.relu(self.bn2(self.conv2(state1)))
		state1 = self.pool(state1)
		state1 = F.relu(self.bn3(self.conv3(state1)))
		state1 = self.pool(state1) ## batch_size x 64 x 16 x 16
		#state1 = F.relu(self.bn4(self.conv4(state1)))

		state2 = F.relu(self.bn1_2(self.conv1_2(state2)))
		state2 = self.pool(state2)
		state2 = F.relu(self.bn2_2(self.conv2_2(state2)))
		state2 = self.pool(state2)
		state2 = F.relu(self.bn3_2(self.conv3_2(state2)))
		state2 = self.pool(state2) ## batch_size x 64 x 16 x 16
		#state2 = F.relu(self.bn4_2(self.conv4_2(state2)))

		state = torch.cat((state1, state2), 1)
		#print('state.shape = {}'.format(state.shape))
		state = F.relu(self.bn5(self.conv5(state)))
		#print('state.shape = {}'.format(state.shape))

		state = state.reshape(state.size(0), -1)
		#print('state.shape = {}'.format(state.shape))
		return state

class Perception_siamese_fusion_new(nn.Module):
	def __init__(self, input_channels=4, h=256, w=256):
		super(Perception_siamese_fusion_new, self).__init__()
		'''
		self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(64)
		self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		## fully convolutional layer
		#self.conv4 = nn.Conv2d(64, 1, kernel_size=1, stride=1)
		#self.bn4 = nn.BatchNorm2d(1)

		self.conv1_2 = nn.Conv2d(1, 2, kernel_size=5, stride=2, padding=2)
		self.bn1_2 = nn.BatchNorm2d(2)
		self.conv2_2 = nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1)
		self.bn2_2 = nn.BatchNorm2d(4)
		self.conv3_2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
		self.bn3_2 = nn.BatchNorm2d(8)
		self.pool_2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		## fully convolutional layer
		#self.conv4_2 = nn.Conv2d(64, 1, kernel_size=1, stride=1)
		#self.bn4_2 = nn.BatchNorm2d(1)

		self.conv5 = nn.Conv2d(72, 1, kernel_size=1, stride=1)
		self.bn5 = nn.BatchNorm2d(1)
		'''
		self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(32)

		self.conv1_2 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)
		self.bn1_2 = nn.BatchNorm2d(16)
		self.conv2_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
		self.bn2_2 = nn.BatchNorm2d(32)

		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(64)

		self.conv4 = nn.Conv2d(64, 1, kernel_size=1, stride=1)
		self.bn4 = nn.BatchNorm2d(1)
		self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

	def forward(self, state):
		assert state.shape[1]==3
		state1 = state[:, :2, :, :]
		state2 = state[:, 2, :, :].unsqueeze(1)
		#print('state2.shape = {}'.format(state2.shape))

		'''
		state1 = F.relu(self.bn1(self.conv1(state1)))
		state1 = self.pool(state1)
		state1 = F.relu(self.bn2(self.conv2(state1)))
		state1 = self.pool(state1)
		state1 = F.relu(self.bn3(self.conv3(state1)))
		state1 = self.pool(state1) ## batch_size x 64 x 16 x 16
		#state1 = F.relu(self.bn4(self.conv4(state1)))

		state2 = F.relu(self.bn1_2(self.conv1_2(state2)))
		state2 = self.pool(state2)
		state2 = F.relu(self.bn2_2(self.conv2_2(state2)))
		state2 = self.pool(state2)
		state2 = F.relu(self.bn3_2(self.conv3_2(state2)))
		state2 = self.pool(state2) ## batch_size x 64 x 16 x 16
		#state2 = F.relu(self.bn4_2(self.conv4_2(state2)))

		state = torch.cat((state1, state2), 1)
		#print('state.shape = {}'.format(state.shape))
		state = F.relu(self.bn5(self.conv5(state)))
		#print('state.shape = {}'.format(state.shape))
		'''
		state1 = F.relu(self.bn1(self.conv1(state1)))
		state1 = self.pool(state1)
		state1 = F.relu(self.bn2(self.conv2(state1)))
		state1 = self.pool(state1)

		state2 = F.relu(self.bn1_2(self.conv1_2(state2)))
		state2 = self.pool(state2)
		state2 = F.relu(self.bn2_2(self.conv2_2(state2)))
		state2 = self.pool(state2)
		#print('state2.shape = {}'.format(state2.shape))

		state = torch.cat((state1, state2), 1)
		state = F.relu(self.bn3(self.conv3(state)))
		#print('state.shape = {}'.format(state.shape))
		state = self.pool(state) ## batch_size x 64 x 16 x 16
		state = F.relu(self.bn4(self.conv4(state)))
		#print('state.shape = {}'.format(state.shape))

		state = state.reshape(state.size(0), -1)
		#print('state.shape = {}'.format(state.shape))
		return state

class Perception_triplet(nn.Module):
	def __init__(self, input_channels=4, h=256, w=256):
		super(Perception_triplet, self).__init__()
		self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(32)
		self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		## fully convolutional layer
		self.conv4 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
		self.bn4 = nn.BatchNorm2d(1)

		self.conv1_2 = nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2)
		self.bn1_2 = nn.BatchNorm2d(16)
		self.conv2_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
		self.bn2_2 = nn.BatchNorm2d(32)
		self.conv3_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.bn3_2 = nn.BatchNorm2d(32)
		self.pool_2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		## fully convolutional layer
		self.conv4_2 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
		self.bn4_2 = nn.BatchNorm2d(1)

		self.conv1_3 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)
		self.bn1_3 = nn.BatchNorm2d(16)
		self.conv2_3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
		self.bn2_3 = nn.BatchNorm2d(32)
		self.conv3_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.bn3_3 = nn.BatchNorm2d(32)
		self.pool_3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		## fully convolutional layer
		self.conv4_3 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
		self.bn4_3 = nn.BatchNorm2d(1)

	def forward(self, state):
		assert state.shape[1]==5
		state1 = state[:, :2, :, :]
		state2 = state[:, 2:4, :, :]
		state3 = state[:, 4, :, :].unsqueeze(1)
		#print('state3.shape = {}'.format(state3.shape))

		state1 = F.relu(self.bn1(self.conv1(state1)))
		state1 = self.pool(state1)
		state1 = F.relu(self.bn2(self.conv2(state1)))
		state1 = self.pool(state1)
		state1 = F.relu(self.bn3(self.conv3(state1)))
		state1 = self.pool(state1) ## num_steps x 32 x 3 x 3
		state1 = F.relu(self.bn4(self.conv4(state1)))
		state1 = state1.reshape(state1.size(0), -1)

		state2 = F.relu(self.bn1_2(self.conv1_2(state2)))
		state2 = self.pool(state2)
		state2 = F.relu(self.bn2_2(self.conv2_2(state2)))
		state2 = self.pool(state2)
		state2 = F.relu(self.bn3_2(self.conv3_2(state2)))
		state2 = self.pool(state2) ## num_steps x 32 x 3 x 3
		state2 = F.relu(self.bn4_2(self.conv4_2(state2)))
		state2 = state2.reshape(state2.size(0), -1)

		state3 = F.relu(self.bn1_3(self.conv1_3(state3)))
		state3 = self.pool(state3)
		state3 = F.relu(self.bn2_3(self.conv2_3(state3)))
		state3 = self.pool(state3)
		state3 = F.relu(self.bn3_3(self.conv3_3(state3)))
		state3 = self.pool(state3) ## num_steps x 32 x 3 x 3
		state3 = F.relu(self.bn4_3(self.conv4_3(state3)))
		state3 = state3.reshape(state3.size(0), -1)

		state = torch.cat((state1, state2, state3), 1)
		#print('state.shape = {}'.format(state.shape))
		return state

class DQN_OVERLAP_Controller(nn.Module):
	## Actor's input_size is different from Critic's
	def __init__(self, perception_module, output_size, input_size=256, input_channels=2):
		super(DQN_OVERLAP_Controller, self).__init__()
		self.perception = perception_module
		self.linear = nn.Linear(input_size, output_size)

	def forward(self, state):
		state = self.perception(state)
		x = self.linear(state)
		return x

class Perception_overlap_recurrent(nn.Module):
	def __init__(self, input_channels, h=256, w=256):
		super(Perception_overlap_recurrent, self).__init__()
		self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(64)
		self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		## 1 x 1 convolutional layer
		self.conv4 = nn.Conv2d(64, 1, kernel_size=1, stride=1)
		self.bn4 = nn.BatchNorm2d(1)

	def forward(self, state, batch_size, time_step):
		state = state.view(batch_size*time_step, 2, 256, 256)
		state = F.relu(self.bn1(self.conv1(state)))
		state = self.pool(state)
		state = F.relu(self.bn2(self.conv2(state)))
		state = self.pool(state)
		state = F.relu(self.bn3(self.conv3(state)))
		state = self.pool(state) ## num_steps x 32 x 3 x 3
		state = F.relu(self.bn4(self.conv4(state)))
		state = state.view(batch_size, time_step, 256)
		return state

class DQN_OVERLAP_Recurrent_Controller(nn.Module):
	def __init__(self, perception_module, output_size, input_size=256, input_channels=2):
		super(DQN_OVERLAP_Recurrent_Controller, self).__init__()
		self.output_size = output_size

		self.perception = perception_module
		#self.rnn = nn.LSTM(input_size=input_size, hidden_size=256, num_layers=1, batch_first=True)
		#self.linear = nn.Linear(256, self.output_size)

		self.rnn = nn.RNN(input_size=input_size, hidden_size=256, num_layers=1, batch_first=True)
		self.adv = nn.Linear(256, self.output_size)
		self.val = nn.Linear(256, 1)
		
	def init_hidden_states(self, batch_size):
		h = torch.zeros(1, batch_size, 256).float().to(device)
		#c = torch.zeros(1, batch_size, 256).float().to(device)
		#return h, c
		return h

	#def forward(self, state, hidden_state, cell_state, batch_size, time_step):
	def forward(self, state, hidden_state, batch_size, time_step):
		state = self.perception(state, batch_size, time_step) ## batch_size x timestep x 256

		#lstm_out = self.rnn(state, (hidden_state, cell_state))
		lstm_out = self.rnn(state, hidden_state)

		out = lstm_out[0][:, time_step-1, :]
		h_n = lstm_out[1]
		#print('out.shape = {}'.format(out.shape))
		#h_n = lstm_out[1][0]
		#c_n = lstm_out[1][1]

		'''
		x = self.linear(out)
		#print('x.shape = {}'.format(x.shape))
		x = x.expand(batch_size, self.output_size)
		return x, (h_n, c_n)
		'''
		adv_out = self.adv(out)
		val_out = self.val(out)

		qout = val_out.expand(batch_size, self.output_size) + (adv_out - adv_out.mean(dim=1).unsqueeze(1).expand(batch_size, self.output_size))

		return qout, h_n

class DQN_OVERLAP_Recurrent_Controller_no_perception(nn.Module):
	def __init__(self, output_size, input_size=256, input_channels=2):
		super(DQN_OVERLAP_Recurrent_Controller_no_perception, self).__init__()
		self.output_size = output_size

		#self.rnn = nn.LSTM(input_size=input_size, hidden_size=256, num_layers=1, batch_first=True)
		self.rnn = nn.RNN(input_size=input_size, hidden_size=256, num_layers=1, batch_first=True)
		#self.linear = nn.Linear(256, self.output_size)
		#self.hidden = self.init_hidden_states()

		self.adv = nn.Linear(256, self.output_size)
		self.val = nn.Linear(256, 1)

	def init_hidden_states(self, batch_size):
		h = torch.zeros(1, batch_size, 256).float().to(device)
		#c = torch.zeros(1, batch_size, 256).float().to(device)
		#return h, c
		return h

	#def forward(self, state, hidden_state, cell_state, batch_size, time_step):
	def forward(self, state, hidden_state, batch_size, time_step):
		#lstm_out = self.rnn(state, (hidden_state, cell_state))
		lstm_out = self.rnn(state, hidden_state)
		out = lstm_out[0][:, time_step-1, :]
		h_n = lstm_out[1]
		#print('out.shape = {}'.format(out.shape))
		#h_n = lstm_out[1][0]
		#c_n = lstm_out[1][1]

		'''
		x = self.linear(out)
		print('x.shape = {}'.format(x.shape))
		x = x.expand(batch_size, self.output_size)
		'''

		adv_out = self.adv(out)
		val_out = self.val(out)

		qout = val_out.expand(batch_size, self.output_size) + (adv_out - adv_out.mean(dim=1).unsqueeze(1).expand(batch_size, self.output_size))
		return qout, h_n#(h_n, c_n)

class DQN_OVERLAP_Recurrent_Controller_no_perception_dense(nn.Module):
	def __init__(self, output_size, input_size=256, input_channels=2):
		super(DQN_OVERLAP_Recurrent_Controller_no_perception_dense, self).__init__()
		self.output_size = output_size

		#self.rnn = nn.LSTM(input_size=input_size, hidden_size=256, num_layers=1, batch_first=True)
		self.rnn = nn.RNN(input_size=input_size, hidden_size=256, num_layers=1, batch_first=True)
		self.linear = nn.Linear(256, self.output_size)

	def init_hidden_states(self, batch_size):
		h = torch.zeros(1, batch_size, 256).float().to(device)
		#c = torch.zeros(1, batch_size, 256).float().to(device)
		#return h, c
		return h

	#def forward(self, state, hidden_state, cell_state, batch_size):
	def forward(self, state, hidden_state, batch_size):
		#lstm_out = self.rnn(state, (hidden_state, cell_state))
		lstm_out = self.rnn(state, hidden_state)
		out = lstm_out[0][:, :, :]
		h_n = lstm_out[1]
		#h_n = lstm_out[1][0]
		#c_n = lstm_out[1][1]

		x = self.linear(out)
		#print('x.shape = {}'.format(x.shape))
		#x = x.expand(batch_size, self.output_size)
		return x, h_n
		#return x, (h_n, c_n)

	#def forward_packed(self, state, hidden_state, cell_state, lengths):
	def forward_packed(self, state, hidden_state, lengths):
		
		state = pack_padded_sequence(state, lengths, batch_first=True)
		lstm_out = self.rnn(state, hidden_state)
		#lstm_out = self.rnn(state, (hidden_state, cell_state))
		out = lstm_out[0]
		h_n = lstm_out[1]
		#h_n = lstm_out[1][0]
		#c_n = lstm_out[1][1]

		# 'out[0]' takes the output from the PackedSequence
		unpacked_out, _ = pad_packed_sequence(out, batch_first=True)
		#print('unpacked_out.shape: {}'.format(unpacked_out.shape))
		#print('unpacked_out: {}'.format(unpacked_out))
		x = self.linear(unpacked_out)
		#print('x.shape = {}'.format(x.shape))
		#x = x.expand(batch_size, self.output_size)
		return x, h_n
		#return x, (h_n, c_n)


class DDPG_Critic(nn.Module):
	def __init__(self, perception_module, hidden_size, output_size, input_size=257):
		super(DDPG_Critic, self).__init__()
		self.perception = perception_module

		self.linear1 = nn.Linear(input_size, hidden_size)
		self.linear2 = nn.Linear(hidden_size, output_size)

		#self.bn1 = nn.BatchNorm1d(hidden_size)

	def forward(self, state, action):
		state = self.perception(state)
		x = torch.cat([state, action], 1)
		#x = F.relu(self.bn1(self.linear1(x)))
		x = F.relu(self.linear1(x))
		x = self.linear2(x)
		#print('critic.x.shape = {}'.format(x.shape))
		return x

class DDPG_Actor(nn.Module):
	## Actor's input_size is different from Critic's
	def __init__(self, perception_module, hidden_size, output_size, input_size=256):
		super(DDPG_Actor, self).__init__()
		self.perception = perception_module
		self.linear1 = nn.Linear(input_size, hidden_size)
		self.linear2 = nn.Linear(hidden_size, output_size)
		#self.bn1 = nn.BatchNorm1d(hidden_size)

	def forward(self, state):
		state = self.perception(state)
		#x = F.relu(self.bn1(self.linear1(state)))
		x = F.relu(self.linear1(state))
		x = torch.tanh(self.linear2(x))
		#print('actor.x.shape = {}'.format(x.shape))
		return x