# Nom de l'exécutable
TARGET = main

# Compilateur
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17 `pkg-config --cflags opencv4`
LDFLAGS  = `pkg-config --libs opencv4`

# Fichiers sources
SRCS = main.cpp Point3D.cpp Converter.cpp

# Fichiers objets générés automatiquement
OBJS = $(SRCS:.cpp=.o)

# Cible par défaut
all: $(TARGET)

# Comment construire l'exécutable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Comment construire chaque .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Nettoyage
clean:
	rm -f $(OBJS) $(TARGET)

# Pour éviter les conflits avec des fichiers du même nom
.PHONY: all clean