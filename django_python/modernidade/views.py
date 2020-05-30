#from django.shortcuts import render
#from .models import Post

#def post_list(request):
    #return render(request, 'modernidade/post_list.html', {})

from django.views.generic import TemplateView
from .models import Cliente_curso, Cliente
from  django.shortcuts import render

class IndexView(TemplateView):
    template_name = 'index.html'

    def get_context_data(self, **kwargs):
        context = super(IndexView, self).get_context_data(**kwargs)
        context['Cliente_curso'] = Cliente_curso.objects.all()
        return context


def database(request):
    cliente = Cliente.objects.all()    
    context = {
        'cliente': cliente
    }

    return render(request, 'database.html',context)



#def client_view(request):
    #obj = Cliente.objects.all()
    #context = {
        #'id': obj.id,
        #'nome': obj.nome,
     #   'telefone': obj.telefone

    #}
    #return request





