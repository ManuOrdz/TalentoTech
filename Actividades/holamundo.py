from abc import ABC

class Saludo(ABC):
    def __init__(self) -> str:
        self.saludo = f'hola mundo!'
        pass
    
    def obtener_saludo(self):
        print(self.saludo)
    
if __name__ == '__main__':
    Saludo().obtener_saludo()