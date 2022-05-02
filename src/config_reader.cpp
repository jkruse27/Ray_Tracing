#include "config_reader.hpp"

std::shared_ptr<Scene> read_scene_from(const char* config_file){
	std::string text;
	
	std::fstream file(config_file, std::fstream::in);

    point3 lower_left_corner;
    std::vector<std::shared_ptr<Shape>> objects;

    Configs config;

    config.aspect_ratio = 16/9;
    config.altura = 400;
    config.largura = 710;
    config.viewport_height = 2.0;
    config.viewport_width = config.aspect_ratio * config.viewport_height;
    config.focal_length = 1.0;

    config.origin = point3(0, 0, 0);
    config.horizontal = vec3(config.viewport_width, 0, 0);
    config.vertical = vec3(0, config.viewport_height, 0);

	while(getline(file, text)){ 
		if(text.size() == 0 || text.at(0) == '#')
			continue;
		else if(text.find("aspect_ratio") != std::string::npos)
			config.aspect_ratio = get_float(text);
		else if(text.find("altura") != std::string::npos)
			config.altura = get_int(text);
		else if(text.find("largura") != std::string::npos)
			config.largura = get_int(text);
		else if(text.find("viewport_height") != std::string::npos)
			config.viewport_height = get_float(text);
		else if(text.find("viewport_width") != std::string::npos)
			config.viewport_width = get_float(text);
		else if(text.find("focal_length") != std::string::npos)
			config.focal_length = get_float(text);
		else if(text.find("origin") != std::string::npos)
			config.origin = get_vec3(text);
		else if(text.find("horizontal") != std::string::npos)
			config.horizontal = get_vec3(text);
		else if(text.find("vertical") != std::string::npos)
			config.vertical = get_vec3(text);
        else if(text.find("sphere") != std::string::npos)
			objects.push_back(get_sphere(text));
        else if(text.find("plane") != std::string::npos)
			objects.push_back(get_plane(text));
	}

    config.lower_left_corner = config.origin - config.horizontal/2 - config.vertical/2 - vec3(0, 0, config.focal_length);

    std::shared_ptr<Camera> camera(new Camera(config.viewport_height,
                                              config.viewport_width,
                                              config.focal_length,
                                              config.origin,
                                              config.horizontal,
                                              config.vertical,
                                              config.lower_left_corner));

    std::shared_ptr<Scene> cena(new Scene(camera, objects, config.altura, config.largura));

    return cena;
}

int get_int(std::string text){
    std::stringstream ss(text);
    int ret = 0;
    std::string trash = "";
    
    ss >> trash >> trash >> ret;

    return ret;
}

float get_float(std::string text){
    std::stringstream ss(text);
    float ret = 0;
    std::string trash = "";
    
    ss >> trash >> trash >> ret;
    return ret;
}

double get_double(std::string text){
    std::stringstream ss(text);
    double ret = 0;
    std::string trash = "";

    ss >> trash >> trash >> ret;
    return ret;
}

vec3 get_vec3(std::string text){
    std::replace(text.begin(), text.end(), ',', ' ');
    std::replace(text.begin(), text.end(), '(', ' ');
    std::replace(text.begin(), text.end(), ')', ' ');
    std::istringstream ss(text);
    double x, y, z;
    std::string trash = "";

    ss  >> trash >> trash >> x >> y >> z;
    return vec3(x,y,z); 
}

color get_color(std::string text){
    std::replace(text.begin(), text.end(), ',', ' ');
    std::replace(text.begin(), text.end(), '(', ' ');
    std::replace(text.begin(), text.end(), ')', ' ');

    std::istringstream ss(text);
    double x, y, z;
    std::string trash = "";
    
    ss  >> trash >> trash >> x >> y >> z;
    return color(x,y,z); 
}

std::string get_string(std::string text){
	std::size_t last_num = text.find_last_of(" ="); 
	return text.substr(last_num+1, text.size()-last_num);
}

std::shared_ptr<Sphere> get_sphere(std::string text){
    std::replace(text.begin(), text.end(), ',', ' ');
    std::replace(text.begin(), text.end(), '(', ' ');
    std::replace(text.begin(), text.end(), ')', ' ');

    std::istringstream ss(text);
    std::string trash = "";
    double a,b,c,radius;
    ss >> trash >> trash >> a >> b >> c >> radius; 
    point3 center = point3(a,b,c);

    std::shared_ptr<Material> material;

    if(text.find("metal") != std::string::npos){
        double r, g, b, fuzz;
        ss >> trash >> r >> g >> b >> fuzz;
        material = std::make_shared<Metal>(color(r,g,b), fuzz);
    }else if(text.find("opaque") != std::string::npos){
        double r, g, b;
        ss >> trash >> r >> g >> b;

        material = std::make_shared<Opaque>(color(r,g,b));
    }else if(text.find("glass") != std::string::npos){
        double r, g, b, index;
        ss >> trash >> r >> g >> b >> index;
        material = std::make_shared<Glass>(color(r,g,b), index);
    }

    std::shared_ptr<Sphere> sphere(new Sphere(center, radius, material));
    return sphere;
}

std::shared_ptr<Plane> get_plane(std::string text){
    std::replace(text.begin(), text.end(), ',', ' ');
    std::replace(text.begin(), text.end(), '(', ' ');
    std::replace(text.begin(), text.end(), ')', ' ');

    std::istringstream ss(text);
    std::string trash = "";

    double x1,x2,x3,y1,y2,y3,z1,z2,z3, width, height;
    ss >> trash >> trash >> x1 >> y1 >> z1 >> x2 >> y2 >> z2 >> x3 >> y3 >> z3 >> width >> height; 
    point3 center = point3(x1,y1,z1);
    vec3 u = vec3(x2,y2,z2);
    vec3 v = vec3(x3,y3,z3);

    std::shared_ptr<Material> material;

    if(text.find("metal") != std::string::npos){
        double r, g, b, fuzz;
        ss >> trash >> r >> g >> b >> fuzz;
        material = std::make_shared<Metal>(color(r,g,b), fuzz);
    }else if(text.find("opaque") != std::string::npos){
        double r, g, b;
        ss >> trash >> r >> g >> b;
        material = std::make_shared<Opaque>(color(r,g,b));
    }else if(text.find("glass") != std::string::npos){
        double r, g, b, index;
        ss >> trash >> r >> g >> b >> index;
        material = std::make_shared<Glass>(color(r,g,b), index);
    }

    std::shared_ptr<Plane> plane(new Plane(center, u, v, width, height, material));

    return plane;
}