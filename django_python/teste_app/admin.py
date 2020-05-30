from django.contrib import admin

from .models import Produtos,Clientes
# Register your models here.

admin.site.register(Clientes)
admin.site.register(Produtos)
