# import torch
# 
# from wildfire_simulator.simulators import ForwardBurnSimulator, fire_burn_step
# 
# def test_burn_step():
#     def model(data):
# 
# 
# def test_simulator():
#     def model(data):
#         return data * 2
# 
#     def step(data, model):
#         return model(data)
# 
#     def transform(data):
#         return data / 5
# 
#     # the inverse transform is defined incorrectly so that it isn't transparent
#     def inv_transform(data):
#         return data * 3
# 
#     simulator = ForwardBurnSimulator(
#         data=10,
#         model=model,
#         step=step,
#         transform=transform,
#         inv_transform=inv_transform
#     )
# 
#     assert simulator.run_to(10) == 6144

