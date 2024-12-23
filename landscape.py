import numpy as np
import random
from PIL import Image

# Terrain mapping to PNG images
terrain_images = {
    0: "mountain.png",  # MOUNTAIN
    1: "river.png",     # RIVER
    2: "grass.png",     # GRASS
    3: "rock.png",      # ROCK
    4: "riverstone.png"  # RIVERROCK
}
MOUNTAIN = 0
RIVER = 1
GRASS = 2
ROCK = 3
RIVERSTONE = 4


def load_map(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        print(lines)
        
        landscape = np.array([list(map(int, line.strip())) for line in lines])
    return landscape


def mutate(landscape, mutation_rate=0.2):
    #y, x = random.randint(0, landscape.shape[0]-1), random.randint(0, landscape.shape[1]-1)
    #landscape[y, x] = random.randint(0, 4)
    height, width = landscape.shape
    center_x, center_y = width // 2, height // 2
    half_x, helf_y = center_x//2, center_y//2
    
    rock_radius = (min(height, width))//2 - 3
    
    weights = [1, 20, 1, 1, 5]  
    weight2 = [20, 1, 3, 2, 1]  
    weight3 = [3, 1, 20, 2, 1]  
    #result = random.choices([0, 1, 2, 3, 4], weights=weights, k=1)
    for y in range(height):
        for x in range(width):
            
            if random.random() < mutation_rate:
                if ((y > center_y - helf_y and y < center_y + helf_y) and
                    (x > center_x - half_x and x < center_x + half_x)):
                    landscape[y, x] = random.choices([0, 1, 2, 3, 4], weights=weights, k=1)[0]
                elif((y < center_y - rock_radius or y > center_y + rock_radius) and
                    (x < center_x - rock_radius or x > center_x + rock_radius)):
                    #landscape[y, x] = random.choices([0, 1, 2, 3, 4], weights=weight2, k=1)[0]
                    landscape[y, x] = random.choices([0, 1, 2, 3, 4], weights=weight2, k=1)[0]
                else:
                    landscape[y, x] = random.choices([0, 1, 2, 3, 4], weights=weight3, k=1)[0]
    return landscape


def initialize_population(size, base_landscape):
    population = []
    for _ in range(size):
        individual = np.copy(base_landscape)
        mutate(individual, 0.1)  
        population.append(individual)
    return population


def select(population, fitness_func):
    weights = [fitness_func(individual) for individual in population]
    total_fitness = sum(weights)
    probabilities = [weight / total_fitness for weight in weights]
    return random.choices(population, probabilities, k=2)  


# Tournament Selection
def tournament_selection(population, fitness, tournament_size=3):
    size = len(population)
    #competitors = random.sample(population, tournament_size)
    competitors = random.sample(range(size), tournament_size)
    
    competitors_fitness = [(fitness[individual][0]+fitness[individual][1]) for individual in competitors]
    
    winner = competitors[competitors_fitness.index(max(competitors_fitness))]
    
    return winner

def crossover(parent1, parent2):
    child = np.copy(parent1)
    y, x = random.randint(0, parent1.shape[0]-1), random.randint(0, parent1.shape[1]-1)
    child[y:, x:] = parent2[y:, x:]  
    return child

def subregion_crossover(population, p1, p2, fitness):
    child1 = np.copy(population[p1])
    child2 = np.copy(population[p2])
    parent1 = population[p1]
    parent2 = population[p2]
    
    height, width = parent1.shape
    subY, subX = height // 3, width // 3
    
    center_x, center_y = height, width // 2
    half_x, half_y = center_x//2, center_y//2

    #subregion_y, subregion_x = random.randint(0, height-subY), random.randint(0, width-subX)
    subregion_y, subregion_x = center_y - half_y, center_x - half_x
    subregion_y2, subregion_x2 = center_y + half_y, center_x + half_x

    if (fitness[p1][0] > fitness[p2][0] and fitness[p1][1] < fitness[p2][1]):
        child2[subregion_y:subregion_y2, subregion_x:subregion_x2] = parent1[subregion_y:subregion_y2, subregion_x:subregion_x2]
        return child2, child2
    elif (fitness[p1][0] < fitness[p2][0] and fitness[p1][1] > fitness[p2][1]):
        child1[subregion_y:subregion_y2, subregion_x:subregion_x2] = parent2[subregion_y:subregion_y2, subregion_x:subregion_x2]
        return child1, child1
        
    return child1, child2


def fitness(landscape, Return="+"):
    score = 0
    river_score = 0
    other_score = 0
   
    height, width = landscape.shape
    center_x, center_y = landscape.shape[1] // 2, landscape.shape[0] // 2
    lake_radius = min(center_x, center_y) // 3
    radius_part = lake_radius // 3
    time = 1
    rock_radius = (min(height, width))//2 - 3
    
    for y in range(height):
        for x in range(width):
            if (y > center_y - lake_radius and y < center_y + lake_radius and
                x > center_x - lake_radius and x < center_x + lake_radius):
                distance = (((y-center_y)**2 + (x-center_x))**2 )**0.5
                distance = int(distance)
                
                if distance < radius_part: time = 50
                elif distance < radius_part*2: time = 40
                else: time = 30
                
                if distance > lake_radius and landscape[y, x] not in [RIVER,RIVERSTONE]:
                    river_score += 10
                elif distance > lake_radius and landscape[y, x] in [RIVER,RIVERSTONE]:
                    river_score -= 10
                elif landscape[y, x] not in [RIVER,RIVERSTONE]:  
                    river_score -= time 
                elif landscape[y, x] in [RIVER,RIVERSTONE]:
                    river_score += time
            elif ((y < center_y - rock_radius or y > center_y + rock_radius ) and
                (x < center_x - rock_radius or x > center_x + rock_radius)):
                if landscape[y, x] == MOUNTAIN:
                    other_score += 1
                elif landscape[y, x] in [RIVER,RIVERSTONE]:
                    other_score -= 2
                else:
                    other_score -= 1
            else:
                if landscape[y, x] == GRASS:
                    other_score += 1
                elif landscape[y, x] in [RIVER,RIVERSTONE]:
                    other_score -= 2
                else:
                    other_score -= 1
           
    if Return == "+":
        return river_score + other_score
    else:
        return river_score, other_score



def run_ga(population_size, generations, base_landscape):
    population = initialize_population(population_size, base_landscape)
    mutate_rate = 0.2
    for generation in range(generations):
        new_population = []
        new_fitness = []
        fitness_total = [ fitness(x, "=") for x in population ]
        for _ in range(population_size // 2):
            #parent1, parent2 = select(population, fitness)
            parent1 = tournament_selection(population, fitness_total)
            parent2 = tournament_selection(population, fitness_total)
            
            child1, child2 = subregion_crossover(population, parent1, parent2, fitness_total)
            mutate(child1)
            mutate(child2)
            new_population.extend([child1, child2])
            
        population.extend(new_population)
        new_fitness = [ fitness(x) for x in population ]
        fitness_total = [ (score[0] + score[1]) for score in fitness_total ]
        fitness_total.extend(new_fitness)
        
        best_10_individuals = sorted(zip(population, fitness_total), key=lambda x: x[1], reverse=True)[:10]
        population = [individual for individual, _ in best_10_individuals]

        
        best_individual = max(population, key=fitness)
        if generation % 100 == 0 :
            print(f"Generation {generation}: Best Fitness = {fitness(best_individual)}")
    #print(len(population))

    return max(population, key=fitness)


def create_image(filepath, landscape, output_file):
    item_image = Image.open(f'{filepath}{terrain_images[1]}')
    item_h, item_w = item_image.size
    
    height, width = landscape.shape
    img_height, img_width = height * item_h, width * item_w
    img = Image.new('RGB', (img_height, img_width))
    for y in range(height):
        img_y = y * item_h
        for x in range(width):
            img_x = x * item_w
            terrain_type = landscape[y, x]
            img.paste(Image.open(f'{filepath}{terrain_images[terrain_type]}'), (img_x, img_y))
    img.save(output_file)


def main(filepath, savepath, run=10):
    for number in range(run):
        # load map
        base_landscape = load_map(f'{filepath}default.map')
        # print(base_landscape)
        # print(base_landscape.shape)

        # run GA
        best_landscape = run_ga(population_size=20, generations=1000, base_landscape=base_landscape)

        # create result
        create_image(filepath, best_landscape, f"{savepath}final_landscape_{number+4}.png")

if __name__ == "__main__":
    main('RPGGame/data/', 'output/', run=6)