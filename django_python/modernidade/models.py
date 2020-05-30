from django.db import models

# Create your models here.


class Base(models.Model):
    criados = models.DateField('Criação', auto_now_add=True)
    modificado = models.DateField('Atualização',auto_now=True)
    ativo = models.BooleanField('Ativo?',default=True)

    class Meta:
        abstract = True




class Cliente_curso(Base):
    id = models.IntegerField('ID', primary_key=True)
    nome = models.CharField('nome', max_length=100)
    telefone = models.CharField('telefone', max_length=100)
    #id = models.IntegerField(primary_key=True)
    #nome = models.CharField(max_length=100)
    #telefone = models.CharField(max_length=100)
    
    class Meta:
        managed = False
        db_table = 'modernidade_cliente'

    
    def __str__(self):
        return self.nome


class Cliente(models.Model):
    id = models.IntegerField(primary_key=True)
    nome = models.CharField(max_length=100, blank=True, null=True)
    telefone = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cliente'

