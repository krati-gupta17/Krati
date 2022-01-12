'''
A template for training your cars using Particle Swarm Optimization
This template is very similar to the one you implemented in Assignment-1
'''

from os import curdir
from template_1_car import MyCar
import numpy as np

class Particle():
    ''' A single partice, part of a swarm for PSO'''
    def __init__(self, tag, hyper_params, weight_size):
        ''' initialize the particle'''
        lo,hi = hyper_params['weight_range']
        self.curr_car = MyCar(id=tag, weights=np.random.uniform(lo,hi,weight_size)) # car is already defined (in imports)
        self.best_car = self.curr_car
        # The particle's velocity (note that this is different from the car's velocity)
        self.vel = np.zeros((weight_size,))

def create_population(weight_size, hyper_params):
    '''function to create swarm population'''
    lo,hi = hyper_params['weight_range']
    return [Particle(i,hyper_params,weight_size) for i in range(hyper_params['population_size'])]


def update_position(hyper_params, population, track): # task 3
    '''Update position of all particles'''
    lo,hi = hyper_params['weight_range']
    pop = population
    for i in range(hyper_params['population_size']):
        pop[i].curr_car.weights = np.maximum(np.minimum(pop[i].curr_car.weights + pop[i].vel, hi), lo)
        if pop[i].curr_car.run(track) > pop[i].best_car.run(track):
            pop[i].best_car = pop[i].curr_car
    return pop

#Update velocity
def update_velocity(hyper_params, population, global_best_car): # task 4
    '''Update velocity of all particles '''
    w, c1, c2 = hyper_params['w'], hyper_params['c1'], hyper_params['c2']
    pop = population
    for i in range(hyper_params['population_size']):
        pop[i].vel = w*pop[i].vel + c1* np.random.random()*(pop[i].best_car.weights - pop[i].curr_car.weights) + c2* np.random.random()*(global_best_car.weights - pop[i].curr_car.weights)
        pop[i].vel /= (w+c1+c2)
    return pop

def PSO(population, track, hyper_params, print_every=10):
    pop = population
    f = [pop[i].curr_car.run(track) for i in range(hyper_params['population_size'])]
    print(f'Iteration 0: [avg: {round(np.mean(f),3)} | best: {round(np.max(f),3)}]')

    global_best_car = max(pop, key=lambda x: x.curr_car.run(track)).curr_car

    for i in range(1,hyper_params['iterations']+1):
        pop = update_velocity(hyper_params, pop, global_best_car)
        pop = update_position(hyper_params, pop, track)

        best_car = max(pop, key=lambda x: x.curr_car.run(track)).curr_car

        if best_car.run(track) > global_best_car.run(track):
            global_best_car = best_car

        if i%print_every == 0:
            f = [p.curr_car.run(track) for p in pop]
            print(f'Iteration {i}: [avg: {round(np.mean(f),3)} | best: {round(np.max(f),3)}]')
    return pop

### ----------------------------------------------------------------------- ###

# These can be freely changed
hyper_params = {
    'w': 10, # PSO parameter - coefficient of particle's inertia
    'c1': 0.1, # PSO parameter - coefficient of particle's local movement
    'c2': 0.2, # PSO parameter - coefficient of particle's local movement
    'weight_range': (-3,3), # Range in which the weights can vary
    'iterations': 15, # Number of iteration -- Original : 15, Optimum value: 18
    'population_size': 30 
    ,# Size of the population -- Original: 30
}

'''
Although weight_size can also be considered as a hyper parameters, it can't be
freely changed as changing this would require a corresponding change in your
car's move() function
'''
weight_size = 8

# create a population of cars
print('Creating the population ...')
pop = create_population(weight_size, hyper_params)

# train the Particle Swarm optimization on different tracks
print('Training on track 1 ...')
track1 = 'tracks/sample_track_1.jpg'
pop = PSO(pop, track1, hyper_params, print_every=3)

print('Training on track 2 ...')
track2 = 'tracks/sample_track_2.jpg'
pop = PSO(pop, track2, hyper_params, print_every=3)

# save the car that runs best on track 2
# best_particle = max(pop, key=lambda x: x.curr_car.run(track2))
# best_particle.curr_car.run(track2, save='best_car_pso_template_track2.gif')
# best_particle.curr_car.save('pso_best_weights')

# continue training on track 3
# car.load('pso_best_weights') # load the saved weights

print('Training on track 3 ...')
track3 = 'tracks/sample_track_3.jpg'
pop = PSO(pop, track3, hyper_params, print_every=3)

print('Training on track 4 ...')
track4 = 'tracks/sample_track_4.jpg'
pop = PSO(pop, track4, hyper_params, print_every=3)

print('Training on track 5 ...')
track5 = 'tracks/sample_track_5.jpg'
pop = PSO(pop, track5, hyper_params, print_every=3)

print('Training on track 6 ...')
track6 = 'tracks/sample_track_6.jpg'
pop = PSO(pop, track6, hyper_params, print_every=3)

print('Training on track 7 ...')
track7 = 'tracks/sample_track_7.jpg'
pop = PSO(pop, track7, hyper_params, print_every=3)

print('Training on track 8 ...')
track8 = 'tracks/sample_track_8.jpg'
pop = PSO(pop, track8, hyper_params, print_every=3)


print('Training on track 9 ...')
track9 = 'tracks/sample_track_9.jpg'
pop = PSO(pop, track9, hyper_params, print_every=3)

# print('Training on track 10 ...')
# track10 = 'tracks/sample_track_10.jpg'
# pop = PSO(pop, track10, hyper_params, print_every=3)

# save the car that runs best on all tracks combined
best_particle = max(pop, key=lambda x: x.curr_car.run(track1) + x.curr_car.run(track2) + x.curr_car.run(track3) + x.curr_car.run(track4) +
x.curr_car.run(track5) + x.curr_car.run(track6) + x.curr_car.run(track7) + x.curr_car.run(track8)
+ x.curr_car.run(track9))
f1 = best_particle.curr_car.run(track1, save='best_car_pso_template_track_1.gif')
f2 = best_particle.curr_car.run(track2, save='best_car_pso_template_track_2.gif')
f3 = best_particle.curr_car.run(track3, save='best_car_pso_template_track_3.gif')
f4 = best_particle.curr_car.run(track4, save='best_car_pso_template_track_4.gif')
f5 = best_particle.curr_car.run(track5, save='best_car_pso_template_track_5.gif')
print(f'Overall fitness: {f1+f2+f3+f4+f5} = {f1} + {f2} + {f3} +{f4} + {f5}')
f6 = best_particle.curr_car.run(track6, save='best_car_pso_template_track_6.gif')
f7 = best_particle.curr_car.run(track7, save='best_car_pso_template_track_7.gif')
f8 = best_particle.curr_car.run(track8, save='best_car_pso_template_track_8.gif')
f9 = best_particle.curr_car.run(track9, save='best_car_pso_template_track_9.gif')
print(f'Overall fitness: {f1+f2+f3+f4+f5+f6+f7+f8+f9} = {f1} + {f2} + {f3} +{f4} + {f5} + {f6} + {f7}+ {f8} + {f9}')
best_particle.curr_car.save(file='pso_best_weights')
