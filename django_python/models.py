# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class AbreFechaCaixa(models.Model):
    codigo = models.AutoField(primary_key=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador', blank=True, null=True)
    a_data = models.DateField(blank=True, null=True)
    a_hora = models.TimeField(blank=True, null=True)
    a_dinheiro = models.DecimalField(max_digits=15, decimal_places=3, blank=True, null=True)
    a_cheque = models.DecimalField(max_digits=15, decimal_places=3, blank=True, null=True)
    a_outro_vl = models.DecimalField(max_digits=15, decimal_places=3, blank=True, null=True)
    f_data = models.DateField(blank=True, null=True)
    f_hora = models.TimeField(blank=True, null=True)
    f_dinheiro = models.DecimalField(max_digits=15, decimal_places=3, blank=True, null=True)
    f_cheque = models.DecimalField(max_digits=15, decimal_places=3, blank=True, null=True)
    f_outro_vl = models.DecimalField(max_digits=15, decimal_places=3, blank=True, null=True)
    fechado = models.CharField(max_length=1, blank=True, null=True)
    exportou = models.BooleanField(blank=True, null=True)
    usuario_abertura = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario_abertura')
    usuario_fechamento = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario_fechamento', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'abre_fecha_caixa'


class Acessos(models.Model):
    form = models.CharField(max_length=-1, blank=True, null=True)
    tag = models.IntegerField(blank=True, null=True)
    data = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'acessos'


class Agenda(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nome = models.CharField(max_length=50, blank=True, null=True)
    telefone = models.CharField(max_length=14, blank=True, null=True)
    fax = models.CharField(max_length=14, blank=True, null=True)
    email = models.CharField(max_length=50, blank=True, null=True)
    url = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'agenda'


class AgroGta(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_item_venda = models.ForeignKey('ItensOrdemServico', models.DO_NOTHING, db_column='cd_item_venda', blank=True, null=True)
    cd_item_compra = models.ForeignKey('ItensNotaCompra', models.DO_NOTHING, db_column='cd_item_compra', blank=True, null=True)
    numero = models.IntegerField(blank=True, null=True)
    tipo = models.CharField(max_length=1, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'agro_gta'


class Alterados(models.Model):
    produto = models.OneToOneField('Produtos', models.DO_NOTHING, db_column='produto', primary_key=True)
    data = models.DateField(blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    hora = models.TimeField()

    class Meta:
        managed = False
        db_table = 'alterados'


class AlteradosClientes(models.Model):
    cliente = models.OneToOneField('Clientes', models.DO_NOTHING, db_column='cliente', primary_key=True)
    data = models.DateField(blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    hora = models.TimeField()

    class Meta:
        managed = False
        db_table = 'alterados_clientes'


class AlteradosFornecedores(models.Model):
    fornecedor = models.OneToOneField('Fornecedores', models.DO_NOTHING, db_column='fornecedor', primary_key=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)
    data = models.DateField()
    hora = models.TimeField()

    class Meta:
        managed = False
        db_table = 'alterados_fornecedores'


class Apagar(models.Model):
    codigo = models.AutoField(primary_key=True)
    vencimento = models.DateField(blank=True, null=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    tipo_documento = models.CharField(max_length=2, blank=True, null=True)
    documento = models.CharField(max_length=20, blank=True, null=True)
    data_emissao = models.DateField(blank=True, null=True)
    data_lancamento = models.DateField(blank=True, null=True)
    hora_lancamento = models.TimeField(blank=True, null=True)
    historico = models.ForeignKey('Historicos', models.DO_NOTHING, db_column='historico', blank=True, null=True)
    complemento = models.CharField(max_length=200, blank=True, null=True)
    fornecedor = models.ForeignKey('Fornecedores', models.DO_NOTHING, db_column='fornecedor', blank=True, null=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador', blank=True, null=True)
    portador = models.ForeignKey('Bancos', models.DO_NOTHING, db_column='portador', blank=True, null=True)
    descricao_adicional = models.CharField(max_length=200, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'apagar'


class ApagarCanceladas(models.Model):
    parcela = models.OneToOneField(Apagar, models.DO_NOTHING, db_column='parcela', primary_key=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)
    motivo = models.ForeignKey('Motivos', models.DO_NOTHING, db_column='motivo', blank=True, null=True)
    complemento = models.CharField(max_length=50, blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'apagar_canceladas'


class ApagarCheques(models.Model):
    apagar = models.ForeignKey('ApagarQuitadas', models.DO_NOTHING, db_column='apagar', blank=True, null=True)
    cheque = models.ForeignKey('Chequespre', models.DO_NOTHING, db_column='cheque', blank=True, null=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'apagar_cheques'


class ApagarDescontos(models.Model):
    parcela = models.OneToOneField('ApagarQuitadas', models.DO_NOTHING, db_column='parcela', primary_key=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'apagar_descontos'


class ApagarDestino(models.Model):
    cd_parcela = models.ForeignKey(Apagar, models.DO_NOTHING, db_column='cd_parcela', blank=True, null=True)
    apagar_quitada = models.ForeignKey('ApagarQuitadas', models.DO_NOTHING, db_column='apagar_quitada', blank=True, null=True)
    id_caixa = models.IntegerField(blank=True, null=True)
    id_banco = models.IntegerField(blank=True, null=True)
    id_juros = models.IntegerField(blank=True, null=True)
    id_multas = models.IntegerField(blank=True, null=True)
    id_desconto = models.IntegerField(blank=True, null=True)
    id_credito = models.IntegerField(blank=True, null=True)
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'apagar_destino'


class ApagarEspecie(models.Model):
    parcela = models.ForeignKey('ApagarQuitadas', models.DO_NOTHING, db_column='parcela')
    especie = models.ForeignKey('Especies', models.DO_NOTHING, db_column='especie', blank=True, null=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'apagar_especie'


class ApagarJuros(models.Model):
    parcela = models.OneToOneField('ApagarQuitadas', models.DO_NOTHING, db_column='parcela', primary_key=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'apagar_juros'


class ApagarMarcadas(models.Model):
    parcela = models.OneToOneField(Apagar, models.DO_NOTHING, db_column='parcela', primary_key=True)

    class Meta:
        managed = False
        db_table = 'apagar_marcadas'


class ApagarMultas(models.Model):
    parcela = models.OneToOneField('ApagarQuitadas', models.DO_NOTHING, db_column='parcela', primary_key=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'apagar_multas'


class ApagarNotaCompra(models.Model):
    apagar = models.OneToOneField(Apagar, models.DO_NOTHING, db_column='apagar', primary_key=True)
    nota = models.ForeignKey('CabecalhoNotaCompra', models.DO_NOTHING, db_column='nota')

    class Meta:
        managed = False
        db_table = 'apagar_nota_compra'
        unique_together = (('apagar', 'nota'),)


class ApagarObservacoes(models.Model):
    parcela = models.OneToOneField(Apagar, models.DO_NOTHING, db_column='parcela', primary_key=True)
    observacao = models.ForeignKey('ObservacoesMovimentos', models.DO_NOTHING, db_column='observacao', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'apagar_observacoes'


class ApagarParciais(models.Model):
    parcela = models.OneToOneField(Apagar, models.DO_NOTHING, db_column='parcela', primary_key=True)
    origem = models.ForeignKey('ApagarQuitadas', models.DO_NOTHING, db_column='origem', blank=True, null=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535)

    class Meta:
        managed = False
        db_table = 'apagar_parciais'


class ApagarQuitadas(models.Model):
    parcela = models.OneToOneField(Apagar, models.DO_NOTHING, db_column='parcela', primary_key=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)
    valor_pago = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    data_pgto = models.DateField(blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'apagar_quitadas'


class ArpaTarefas(models.Model):
    codigo = models.IntegerField()
    codigotarefa = models.CharField(primary_key=True, max_length=-1)
    tarefa = models.CharField(max_length=-1, blank=True, null=True)
    mensagem = models.CharField(max_length=-1, blank=True, null=True)
    dataoperacao = models.DateTimeField()
    executou = models.BooleanField()
    inicioexecucao = models.DateTimeField(blank=True, null=True)
    terminoexecucao = models.DateTimeField(blank=True, null=True)
    mostrar_erro = models.BooleanField()
    parar_erro = models.BooleanField()
    apos_triggers = models.BooleanField()
    versao = models.CharField(max_length=15)

    class Meta:
        managed = False
        db_table = 'arpa_tarefas'


class ArpaTarefasPre(models.Model):
    codigo = models.IntegerField()
    codigotarefa = models.CharField(primary_key=True, max_length=-1)
    tarefa = models.CharField(max_length=-1, blank=True, null=True)
    mensagem = models.CharField(max_length=-1, blank=True, null=True)
    dataoperacao = models.DateTimeField()
    executou = models.BooleanField()
    inicioexecucao = models.DateTimeField(blank=True, null=True)
    terminoexecucao = models.DateTimeField(blank=True, null=True)
    mostrar_erro = models.BooleanField()
    parar_erro = models.BooleanField()
    apos_triggers = models.BooleanField()
    versao = models.CharField(max_length=15)

    class Meta:
        managed = False
        db_table = 'arpa_tarefas_pre'


class Art(models.Model):
    art = models.CharField(primary_key=True, max_length=-1)
    datacompra = models.DateField(blank=True, null=True)
    quantidade_comprada = models.IntegerField()
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    codigo_inicial = models.IntegerField(blank=True, null=True)
    inativo = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'art'


class ArtItensOsMarcados(models.Model):
    codigo = models.OneToOneField('ItensOrdemServico', models.DO_NOTHING, db_column='codigo', primary_key=True)

    class Meta:
        managed = False
        db_table = 'art_itens_os_marcados'


class ArtNumeroReceitaOs(models.Model):
    art = models.ForeignKey(Art, models.DO_NOTHING, db_column='art')
    item_os = models.OneToOneField('ItensOrdemServico', models.DO_NOTHING, db_column='item_os', primary_key=True)
    codigo_art = models.IntegerField(blank=True, null=True)
    receita_impressa = models.BooleanField(blank=True, null=True)
    data_receita = models.DateField()
    vendedor = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='vendedor', blank=True, null=True)
    data_informada = models.DateField(blank=True, null=True)
    lei_impressao = models.CharField(max_length=-1, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'art_numero_receita_os'


class Atividades(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nome = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'atividades'


class AuthGroup(models.Model):
    name = models.CharField(unique=True, max_length=150)

    class Meta:
        managed = False
        db_table = 'auth_group'


class AuthGroupPermissions(models.Model):
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)
    permission = models.ForeignKey('AuthPermission', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_group_permissions'
        unique_together = (('group', 'permission'),)


class AuthPermission(models.Model):
    name = models.CharField(max_length=255)
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING)
    codename = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'auth_permission'
        unique_together = (('content_type', 'codename'),)


class AuthUser(models.Model):
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField(blank=True, null=True)
    is_superuser = models.BooleanField()
    username = models.CharField(unique=True, max_length=150)
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=150)
    email = models.CharField(max_length=254)
    is_staff = models.BooleanField()
    is_active = models.BooleanField()
    date_joined = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'auth_user'


class AuthUserGroups(models.Model):
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_groups'
        unique_together = (('user', 'group'),)


class AuthUserUserPermissions(models.Model):
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    permission = models.ForeignKey(AuthPermission, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_user_permissions'
        unique_together = (('user', 'permission'),)


class AutorizadosRemoto(models.Model):
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)
    filial = models.ForeignKey('Filiais', models.DO_NOTHING, db_column='filial', blank=True, null=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    cd_item = models.ForeignKey('ItensOrdemServico', models.DO_NOTHING, db_column='cd_item', blank=True, null=True)
    codigo = models.AutoField(primary_key=True)
    data = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'autorizados_remoto'


class Avista(models.Model):
    pedido = models.ForeignKey('CabecalhoOrdemServico', models.DO_NOTHING, db_column='pedido', blank=True, null=True)
    nome = models.CharField(max_length=50, blank=True, null=True)
    cpf_cnpj = models.CharField(max_length=18, blank=True, null=True)
    endereco = models.CharField(max_length=50, blank=True, null=True)
    bairro = models.CharField(max_length=50, blank=True, null=True)
    cep = models.ForeignKey('Cidades', models.DO_NOTHING, db_column='cep', blank=True, null=True)
    municipio = models.CharField(max_length=50, blank=True, null=True)
    fone = models.CharField(max_length=14, blank=True, null=True)
    uf = models.CharField(max_length=2, blank=True, null=True)
    ie = models.CharField(max_length=20, blank=True, null=True)
    codigo = models.AutoField(primary_key=True)
    ped_avulso = models.ForeignKey('CabecaNotaAvulsa', models.DO_NOTHING, db_column='ped_avulso', blank=True, null=True)
    servicos = models.ForeignKey('Servicos', models.DO_NOTHING, db_column='servicos', blank=True, null=True)
    remessa_venda_fora = models.ForeignKey('CabecalhoRemessaVendaFora', models.DO_NOTHING, db_column='remessa_venda_fora', blank=True, null=True)
    orcamento = models.ForeignKey('CabecalhoOrcamento', models.DO_NOTHING, db_column='orcamento', blank=True, null=True)
    retorno_venda_fora = models.ForeignKey('CabecalhoRetornoVendaFora', models.DO_NOTHING, db_column='retorno_venda_fora', blank=True, null=True)
    visita = models.ForeignKey('ControleVisitas', models.DO_NOTHING, db_column='visita', blank=True, null=True)
    condicional = models.ForeignKey('CabecalhoCondicional', models.DO_NOTHING, db_column='condicional', blank=True, null=True)
    devolucao = models.ForeignKey('CabecalhoDevolucao', models.DO_NOTHING, db_column='devolucao', blank=True, null=True)
    produtor_rural = models.BooleanField()
    simples = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'avista'


class Backups(models.Model):
    horario = models.CharField(max_length=5)
    executado = models.DateTimeField(blank=True, null=True)
    executando = models.BooleanField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'backups'


class BackupsConfig(models.Model):
    copias = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'backups_config'


class BackupsCopias(models.Model):
    backup = models.ForeignKey(Backups, models.DO_NOTHING, db_column='backup')
    gerado = models.DateTimeField()
    arquivo = models.CharField(max_length=-1)
    tamanho = models.CharField(max_length=-1)
    tamanho_7z = models.CharField(max_length=-1)

    class Meta:
        managed = False
        db_table = 'backups_copias'


class Balancas(models.Model):
    produto = models.OneToOneField('Produtos', models.DO_NOTHING, db_column='produto', primary_key=True)
    teclado = models.IntegerField(blank=True, null=True)
    posicao = models.IntegerField(blank=True, null=True)
    dias_validade = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'balancas'


class Bancos(models.Model):
    numero = models.AutoField(primary_key=True)
    nome = models.CharField(max_length=50, blank=True, null=True)
    agencia = models.SmallIntegerField(blank=True, null=True)
    telefone = models.CharField(max_length=13, blank=True, null=True)
    num_banco = models.SmallIntegerField(blank=True, null=True)
    modelo_cheque = models.CharField(max_length=3, blank=True, null=True)
    dv_agencia = models.CharField(max_length=2, blank=True, null=True)
    posto = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'bancos'


class Bandeiras(models.Model):
    codigo = models.AutoField(primary_key=True)
    nome = models.CharField(max_length=60, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'bandeiras'


class Beneficio(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    nome = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'beneficio'


class BeneficioCodigo(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    nome = models.CharField(unique=True, max_length=10)
    descricao = models.CharField(max_length=10000)

    class Meta:
        managed = False
        db_table = 'beneficio_codigo'


class BeneficioConfiguracao(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    beneficio = models.ForeignKey(Beneficio, models.DO_NOTHING, db_column='beneficio')
    beneficio_codigo = models.ForeignKey(BeneficioCodigo, models.DO_NOTHING, db_column='beneficio_codigo')
    uf = models.CharField(max_length=5)
    destinatario_tipo = models.IntegerField()
    destinatario_beneficio = models.IntegerField()
    tipo_operacao = models.IntegerField()
    destino_mercadoria = models.IntegerField()
    bonificado = models.IntegerField()
    cst = models.CharField(max_length=4)

    class Meta:
        managed = False
        db_table = 'beneficio_configuracao'


class Boletos(models.Model):
    boleto = models.DecimalField(max_digits=17, decimal_places=0, blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador', blank=True, null=True)
    cancelado = models.CharField(max_length=1)
    parcela = models.ForeignKey('Parcelas', models.DO_NOTHING, db_column='parcela', blank=True, null=True)
    conta = models.ForeignKey('Ctabancarias', models.DO_NOTHING, db_column='conta', blank=True, null=True)
    codigo = models.AutoField(primary_key=True)
    marca = models.CharField(max_length=1, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'boletos'


class BoletosInfLote(models.Model):
    codigo = models.AutoField(primary_key=True)
    lote = models.ForeignKey('BoletosLote', models.DO_NOTHING, db_column='lote')
    aceite = models.IntegerField(blank=True, null=True)
    remessa = models.IntegerField(blank=True, null=True)
    carteira = models.IntegerField(blank=True, null=True)
    cd_movimento = models.IntegerField(blank=True, null=True)
    especie = models.IntegerField(blank=True, null=True)
    tipo_documento = models.IntegerField(blank=True, null=True)
    emite_boleto = models.IntegerField(blank=True, null=True)
    distr_boleto = models.IntegerField(blank=True, null=True)
    protestar = models.IntegerField(blank=True, null=True)
    dia_protesto = models.IntegerField(blank=True, null=True)
    mensagem1 = models.CharField(max_length=40, blank=True, null=True)
    mensagem2 = models.CharField(max_length=40, blank=True, null=True)
    datahora = models.DateTimeField(blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    caminho = models.CharField(max_length=255, blank=True, null=True)
    cnae = models.IntegerField(blank=True, null=True)
    instrucao1 = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'boletos_inf_lote'


class BoletosLote(models.Model):
    codigo = models.AutoField(primary_key=True)
    convenio = models.ForeignKey('ConvenioBancos', models.DO_NOTHING, db_column='convenio')
    datahora = models.DateTimeField(blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    status = models.CharField(max_length=1, blank=True, null=True)
    datahorafechamentocancelamento = models.DateTimeField(blank=True, null=True)
    usuariofechamentocancelamento = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuariofechamentocancelamento', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'boletos_lote'


class BoletosLoteParcelas(models.Model):
    codigo = models.AutoField(primary_key=True)
    lote = models.ForeignKey(BoletosLote, models.DO_NOTHING, db_column='lote')
    parcela = models.ForeignKey('Parcelas', models.DO_NOTHING, db_column='parcela', blank=True, null=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    enviado = models.BooleanField()
    usuarioreenvio = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuarioreenvio', blank=True, null=True)
    datahorareenvio = models.DateTimeField(blank=True, null=True)
    parcela_excluida = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'boletos_lote_parcelas'


class BoletosRetornoLote(models.Model):
    codigo = models.AutoField(primary_key=True)
    convenio = models.ForeignKey('ConvenioBancos', models.DO_NOTHING, db_column='convenio')
    datahora = models.DateTimeField(blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    status = models.CharField(max_length=1, blank=True, null=True)
    datahorafechamentocancelamento = models.DateTimeField(blank=True, null=True)
    usuariofechamentocancelamento = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuariofechamentocancelamento', blank=True, null=True)
    numeroremessa = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'boletos_retorno_lote'


class BoletosRetornoLoteParcelas(models.Model):
    codigo = models.AutoField(primary_key=True)
    lote = models.ForeignKey(BoletosRetornoLote, models.DO_NOTHING, db_column='lote')
    parcela = models.ForeignKey('Parcelas', models.DO_NOTHING, db_column='parcela', blank=True, null=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    tiporegistro = models.CharField(max_length=100, blank=True, null=True)
    ocorrenciaoriginal = models.CharField(max_length=100, blank=True, null=True)
    datapagamento = models.DateField(blank=True, null=True)
    juros = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    multa = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    desconto = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    valortarifa = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    diferenca = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    nossonumero = models.CharField(max_length=20, blank=True, null=True)
    valorrecebido = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    motivorejeicao = models.CharField(max_length=1000, blank=True, null=True)
    datacredito = models.DateField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'boletos_retorno_lote_parcelas'


class BoletosRetornoProtesto(models.Model):
    codigo = models.AutoField(primary_key=True)
    parcela = models.ForeignKey('Parcelas', models.DO_NOTHING, db_column='parcela')
    data_protesto = models.DateField(blank=True, null=True)
    datahora = models.DateTimeField(blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'boletos_retorno_protesto'


class BoletosRetornoProtestoSustado(models.Model):
    codigo = models.AutoField(primary_key=True)
    parcela = models.ForeignKey('Parcelas', models.DO_NOTHING, db_column='parcela')
    data_sustado = models.DateField(blank=True, null=True)
    datahora = models.DateTimeField(blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'boletos_retorno_protesto_sustado'


class BoletosRetornos(models.Model):
    codigo = models.AutoField(primary_key=True)
    parcela = models.ForeignKey('Parcelas', models.DO_NOTHING, db_column='parcela')
    mensagem = models.CharField(max_length=-1, blank=True, null=True)
    datahora = models.DateTimeField(blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    remessa = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'boletos_retornos'


class Bombas(models.Model):
    codigo = models.AutoField(primary_key=True)
    bomba = models.CharField(max_length=20, blank=True, null=True)
    bico = models.CharField(max_length=20, blank=True, null=True)
    numero = models.CharField(max_length=20, blank=True, null=True)
    data = models.DateField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'bombas'


class CabecaNotaAvulsa(models.Model):
    pedido = models.AutoField(primary_key=True)
    data = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)
    cliente = models.ForeignKey('Clientes', models.DO_NOTHING, db_column='cliente', blank=True, null=True)
    vendedor = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='vendedor', blank=True, null=True)
    transportadora = models.ForeignKey('Fornecedores', models.DO_NOTHING, db_column='transportadora', blank=True, null=True)
    frete = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    seguro = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    outrasdespesas = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    status = models.CharField(max_length=1, blank=True, null=True)
    data_fechamento = models.DateField(blank=True, null=True)
    hora_fechamento = models.TimeField(blank=True, null=True)
    cod_observacao = models.ForeignKey('ObservacoesMovimentos', models.DO_NOTHING, db_column='cod_observacao', blank=True, null=True)
    data_cancelamento = models.DateField(blank=True, null=True)
    hora_cancelamento = models.TimeField(blank=True, null=True)
    usuario_cancelamento = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario_cancelamento', blank=True, null=True)
    finalidade = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'cabeca_nota_avulsa'


class CabecaPreco(models.Model):
    codigo = models.AutoField(primary_key=True)
    descricao = models.CharField(max_length=100)
    data_lancamento = models.DateField()
    vigencia = models.DateField()
    comissaovendedor = models.DecimalField(max_digits=65535, decimal_places=65535)
    comissaoprospectador = models.DecimalField(max_digits=65535, decimal_places=65535)
    data_alteracao = models.DateTimeField()
    envia_palm = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'cabeca_preco'


class CabecaPrecoDesativaPalm(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_tabela = models.IntegerField(blank=True, null=True)
    data = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'cabeca_preco_desativa_palm'


class CabecalhoAlteracaoTributacao(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    nome = models.CharField(max_length=60)

    class Meta:
        managed = False
        db_table = 'cabecalho_alteracao_tributacao'


class CabecalhoCondicional(models.Model):
    pedido = models.AutoField(primary_key=True)
    cd_cliente = models.ForeignKey('Clientes', models.DO_NOTHING, db_column='cd_cliente', blank=True, null=True)
    data = models.DateField()
    data_fechamento = models.DateTimeField(blank=True, null=True)
    data_cancelamento = models.DateTimeField(blank=True, null=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    status = models.CharField(max_length=1)
    cd_obs = models.ForeignKey('ObservacoesMovimentos', models.DO_NOTHING, db_column='cd_obs', blank=True, null=True)
    operador_cancel = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_condicional'


class CabecalhoConferenciaEstoque(models.Model):
    codigo = models.AutoField(primary_key=True)
    vendedor = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='vendedor', blank=True, null=True)
    dataconferencia = models.DateField()
    status = models.CharField(max_length=1)
    nmotivo = models.ForeignKey('Motivos', models.DO_NOTHING, db_column='nmotivo', blank=True, null=True)
    motivo = models.CharField(max_length=100, blank=True, null=True)
    descricao = models.CharField(max_length=60)
    localizacao = models.CharField(max_length=20)
    grupo_inicial = models.IntegerField()
    grupo_final = models.IntegerField()
    subgrupo_inicial = models.IntegerField()
    subgrupo_final = models.IntegerField()
    departamento_inicial = models.IntegerField()
    departamento_final = models.IntegerField()
    classe_inicial = models.IntegerField()
    classe_final = models.IntegerField()
    localizacao_tipo = models.IntegerField()
    datahoraconfirmacao = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_conferencia_estoque'


class CabecalhoDetalhesFormacaoPrecoVenda(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    item_venda = models.ForeignKey('ItensOrdemServico', models.DO_NOTHING, db_column='item_venda', blank=True, null=True)
    item_reserva = models.ForeignKey('ItensOrcamento', models.DO_NOTHING, db_column='item_reserva', blank=True, null=True)
    item_nota_compra = models.ForeignKey('ItensNotaCompra', models.DO_NOTHING, db_column='item_nota_compra', blank=True, null=True)
    padrao = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'cabecalho_detalhes_formacao_preco_venda'


class CabecalhoDevolucao(models.Model):
    pedido = models.AutoField(primary_key=True)
    cd_cliente = models.ForeignKey('Clientes', models.DO_NOTHING, db_column='cd_cliente', blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    status = models.CharField(max_length=1, blank=True, null=True)
    responsavel = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='responsavel', blank=True, null=True)
    observacao = models.ForeignKey('ObservacoesMovimentos', models.DO_NOTHING, db_column='observacao', blank=True, null=True)
    frete = models.DecimalField(max_digits=65535, decimal_places=65535)
    icms = models.DecimalField(max_digits=65535, decimal_places=65535)
    ipi = models.DecimalField(max_digits=65535, decimal_places=65535)
    tipo_frete = models.IntegerField()
    campo = models.BooleanField(blank=True, null=True)
    dataold = models.DateField(blank=True, null=True)
    hora_lancamento = models.TimeField()
    pedido_venda = models.ForeignKey('CabecalhoOrdemServico', models.DO_NOTHING, db_column='pedido_venda', blank=True, null=True)
    data_abertura = models.DateTimeField(blank=True, null=True)
    hora_fechamento = models.TimeField(blank=True, null=True)
    datahoracancelamento = models.DateTimeField(blank=True, null=True)
    operadorcancelamento = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operadorcancelamento', blank=True, null=True)
    motivo_cancelamento = models.ForeignKey('Motivos', models.DO_NOTHING, db_column='motivo_cancelamento', blank=True, null=True)
    complemento_cancelamento = models.CharField(max_length=-1, blank=True, null=True)
    finalidade = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'cabecalho_devolucao'


class CabecalhoDevolucaoCredito(models.Model):
    devolucao = models.OneToOneField(CabecalhoDevolucao, models.DO_NOTHING, db_column='devolucao', primary_key=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_devolucao_credito'


class CabecalhoDevolucaoXml(models.Model):
    codigo = models.OneToOneField(CabecalhoDevolucao, models.DO_NOTHING, db_column='codigo', primary_key=True)
    xml = models.TextField(blank=True, null=True)
    chave = models.CharField(max_length=44)

    class Meta:
        managed = False
        db_table = 'cabecalho_devolucao_xml'


class CabecalhoEnderecosWebservice(models.Model):
    codigo = models.AutoField(primary_key=True)
    nome = models.CharField(max_length=60, blank=True, null=True)
    scan = models.BooleanField(blank=True, null=True)
    codigoscan = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'cabecalho_enderecos_webservice'


class CabecalhoEnderecosWebserviceCte(models.Model):
    codigo = models.AutoField(primary_key=True)
    nome = models.CharField(max_length=60, blank=True, null=True)
    scan = models.BooleanField(blank=True, null=True)
    codigoscan = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'cabecalho_enderecos_webservice_cte'


class CabecalhoEnderecosWebserviceNfce(models.Model):
    codigo = models.AutoField(primary_key=True)
    nome = models.CharField(max_length=60, blank=True, null=True)
    scan = models.BooleanField(blank=True, null=True)
    codigoscan = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'cabecalho_enderecos_webservice_nfce'


class CabecalhoEnderecosWebserviceNfese(models.Model):
    codigo = models.OneToOneField(CabecalhoEnderecosWebservice, models.DO_NOTHING, db_column='codigo', primary_key=True)
    provedor = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_enderecos_webservice_nfese'


class CabecalhoEntrega(models.Model):
    pedido = models.AutoField(primary_key=True)
    data = models.DateField(blank=True, null=True)
    cliente = models.ForeignKey('Clientes', models.DO_NOTHING, db_column='cliente', blank=True, null=True)
    vendedor = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='vendedor', blank=True, null=True)
    status = models.CharField(max_length=1, blank=True, null=True)
    data_cancelamento = models.DateField(blank=True, null=True)
    usuario_cancelamento = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario_cancelamento', blank=True, null=True)
    transportadora = models.ForeignKey('Fornecedores', models.DO_NOTHING, db_column='transportadora', blank=True, null=True)
    data_hora_fechamento = models.DateTimeField(blank=True, null=True)
    hora_cancelamento = models.TimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_entrega'


class CabecalhoMetas(models.Model):
    codigo = models.AutoField(primary_key=True)
    descricao = models.CharField(max_length=50, blank=True, null=True)
    competencia_inicial = models.DateField(blank=True, null=True)
    competencia_final = models.DateField(blank=True, null=True)
    valor = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_metas'


class CabecalhoModeloOrcamento(models.Model):
    ordem = models.IntegerField(primary_key=True)
    nome = models.CharField(max_length=100, blank=True, null=True)
    valor = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    opcao = models.CharField(max_length=1, blank=True, null=True)
    cd_produto = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_modelo_orcamento'


class CabecalhoNotaAvulsaDocrefCupons(models.Model):
    codigo = models.AutoField(primary_key=True)
    nf_avulsa = models.ForeignKey(CabecaNotaAvulsa, models.DO_NOTHING, db_column='nf_avulsa')
    serie = models.CharField(max_length=22)
    caixa = models.IntegerField()
    coo = models.IntegerField()
    data = models.DateField()
    modelo = models.CharField(max_length=2)
    data_operacao = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'cabecalho_nota_avulsa_docref_cupons'


class CabecalhoNotaAvulsaDocrefNotas(models.Model):
    codigo = models.AutoField(primary_key=True)
    nf_avulsa = models.ForeignKey(CabecaNotaAvulsa, models.DO_NOTHING, db_column='nf_avulsa')
    modelo = models.CharField(max_length=4, blank=True, null=True)
    serie = models.CharField(max_length=3, blank=True, null=True)
    data_documento = models.DateField(blank=True, null=True)
    nota = models.IntegerField(blank=True, null=True)
    chave_nota = models.CharField(max_length=44, blank=True, null=True)
    data_operacao = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    operacao = models.IntegerField(blank=True, null=True)
    indicador_proprio_terceiro = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'cabecalho_nota_avulsa_docref_notas'


class CabecalhoNotaAvulsaVendas(models.Model):
    codigo = models.AutoField(primary_key=True)
    avulsa = models.ForeignKey(CabecaNotaAvulsa, models.DO_NOTHING, db_column='avulsa')
    venda = models.ForeignKey('CabecalhoOrdemServico', models.DO_NOTHING, db_column='venda')

    class Meta:
        managed = False
        db_table = 'cabecalho_nota_avulsa_vendas'


class CabecalhoNotaCompra(models.Model):
    codigo = models.AutoField(primary_key=True)
    nota = models.IntegerField(blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    fornecedor = models.ForeignKey('Fornecedores', models.DO_NOTHING, db_column='fornecedor', blank=True, null=True)
    modelo = models.CharField(max_length=2)
    serie = models.CharField(max_length=3)
    total_nota = models.DecimalField(max_digits=65535, decimal_places=65535)
    total_produtos = models.DecimalField(max_digits=65535, decimal_places=65535)
    desconto = models.DecimalField(max_digits=65535, decimal_places=65535)
    data_processamento = models.DateField(blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    hora = models.TimeField(blank=True, null=True)
    status = models.CharField(max_length=1, blank=True, null=True)
    importou_xml = models.BooleanField(blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    funrural = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    observacoes = models.ForeignKey('ObservacoesMovimentos', models.DO_NOTHING, db_column='observacoes', blank=True, null=True)
    chave_nfe = models.CharField(max_length=44)
    capital_social = models.DecimalField(max_digits=65535, decimal_places=65535)
    atualizou_custo_venda = models.BooleanField()
    hora_processamento = models.TimeField(blank=True, null=True)
    codigo_cidade_origem = models.IntegerField()
    codigo_cidade_destino = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'cabecalho_nota_compra'


class CabecalhoNotaCompraAtualizacao(models.Model):
    cd_nota = models.ForeignKey(CabecalhoNotaCompra, models.DO_NOTHING, db_column='cd_nota')
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    data = models.DateField()
    hora = models.TimeField()
    data_processamento = models.DateField()
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_nota_compra_atualizacao'


class CabecalhoNotaCompraAvulsa(models.Model):
    cd_nota = models.OneToOneField(CabecalhoNotaCompra, models.DO_NOTHING, db_column='cd_nota', primary_key=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_nota_compra_avulsa'


class CabecalhoNotaCompraCuponsFiscais(models.Model):
    cd_nota = models.ForeignKey(CabecalhoNotaCompra, models.DO_NOTHING, db_column='cd_nota')
    serie = models.CharField(max_length=22)
    caixa = models.IntegerField()
    coo = models.IntegerField()
    data = models.DateField()
    modelo = models.CharField(max_length=10)
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_nota_compra_cupons_fiscais'


class CabecalhoNotaCompraDespesasExtras(models.Model):
    cd_nota = models.OneToOneField(CabecalhoNotaCompra, models.DO_NOTHING, db_column='cd_nota', primary_key=True)
    documento = models.CharField(max_length=50, blank=True, null=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535)
    desconto = models.DecimalField(max_digits=65535, decimal_places=65535)

    class Meta:
        managed = False
        db_table = 'cabecalho_nota_compra_despesas_extras'


class CabecalhoNotaCompraDocImportacao(models.Model):
    cd_nota = models.OneToOneField(CabecalhoNotaCompra, models.DO_NOTHING, db_column='cd_nota', primary_key=True)
    codigo_importacao = models.IntegerField()
    numero_doc_importacao = models.CharField(max_length=50)
    thc = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    siscomex = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    ii = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    localdesembaraco = models.CharField(max_length=60, blank=True, null=True)
    ufdesembaraco = models.CharField(max_length=2, blank=True, null=True)
    datadesembaraco = models.DateField(blank=True, null=True)
    datadesembaracoaduaneiro = models.DateField(blank=True, null=True)
    dupping = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    seguro = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_nota_compra_doc_importacao'


class CabecalhoNotaCompraEnergiaEletrica(models.Model):
    cd_nota = models.OneToOneField(CabecalhoNotaCompra, models.DO_NOTHING, db_column='cd_nota', primary_key=True)
    total_fornecido = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    data_entrada_saida = models.DateField(blank=True, null=True)
    servico_terceiros = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    despesas_acessorias = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    observacao = models.ForeignKey('Observacoes', models.DO_NOTHING, db_column='observacao', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_nota_compra_energia_eletrica'


class CabecalhoNotaCompraFob(models.Model):
    cd_nota = models.OneToOneField(CabecalhoNotaCompra, models.DO_NOTHING, db_column='cd_nota', primary_key=True)
    conhecimento = models.IntegerField()
    modelo = models.IntegerField()
    serie = models.CharField(max_length=4)
    sub_serie = models.IntegerField()
    data_doc = models.DateField(blank=True, null=True)
    data_saida = models.DateField(blank=True, null=True)
    chave_conhecimento = models.CharField(max_length=44, blank=True, null=True)
    chave_conhecimento_referencia = models.CharField(max_length=44, blank=True, null=True)
    tipo_conhecimento = models.IntegerField()
    valor = models.DecimalField(max_digits=65535, decimal_places=65535)
    desconto = models.DecimalField(max_digits=65535, decimal_places=65535)
    icms = models.DecimalField(max_digits=65535, decimal_places=65535)
    observacao = models.ForeignKey('Observacoes', models.DO_NOTHING, db_column='observacao', blank=True, null=True)
    percentual = models.DecimalField(max_digits=65535, decimal_places=65535)
    total = models.DecimalField(max_digits=65535, decimal_places=65535)
    adicionanabaseicmsst = models.BooleanField()
    xml = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_nota_compra_fob'


class CabecalhoNotaCompraGnre(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_nota = models.ForeignKey(CabecalhoNotaCompra, models.DO_NOTHING, db_column='cd_nota')
    documento_arrecadacao = models.CharField(max_length=500)
    uf_destinatario = models.CharField(max_length=2)
    documento = models.CharField(max_length=500)
    codigo_auten_bancaria = models.CharField(max_length=500, blank=True, null=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535)
    data_vencimento = models.DateField(blank=True, null=True)
    data_pagamento = models.DateField(blank=True, null=True)
    inf_complementares = models.CharField(max_length=1000, blank=True, null=True)
    inclusa_nas_inf_complementares = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'cabecalho_nota_compra_gnre'


class CabecalhoNotaCompraImportacaoXml(models.Model):
    codigo = models.AutoField(primary_key=True)
    chave_nfe = models.CharField(max_length=44)

    class Meta:
        managed = False
        db_table = 'cabecalho_nota_compra_importacao_xml'


class CabecalhoNotaCompraImpostos(models.Model):
    cd_nota = models.OneToOneField(CabecalhoNotaCompra, models.DO_NOTHING, db_column='cd_nota', primary_key=True)
    base_icms = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    valor_icms = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    base_icms_st = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    valor_icms_st = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    ipi = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    pis = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    cofins = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    fcp_icms_valor = models.DecimalField(max_digits=65535, decimal_places=65535)
    fcp_icms_st_valor = models.DecimalField(max_digits=65535, decimal_places=65535)
    desonerado_valor = models.DecimalField(max_digits=20, decimal_places=2)

    class Meta:
        managed = False
        db_table = 'cabecalho_nota_compra_impostos'


class CabecalhoNotaCompraNotasReferenciadas(models.Model):
    cd_nota = models.ForeignKey(CabecalhoNotaCompra, models.DO_NOTHING, db_column='cd_nota')
    serie = models.CharField(max_length=3)
    modelo = models.CharField(max_length=2, blank=True, null=True)
    chave_nota = models.CharField(max_length=44, blank=True, null=True)
    nota = models.IntegerField(blank=True, null=True)
    codigo = models.AutoField(primary_key=True)
    data_documento = models.DateField()

    class Meta:
        managed = False
        db_table = 'cabecalho_nota_compra_notas_referenciadas'


class CabecalhoNotaCompraOutras(models.Model):
    cd_nota = models.OneToOneField(CabecalhoNotaCompra, models.DO_NOTHING, db_column='cd_nota', primary_key=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535)
    icms = models.DecimalField(max_digits=65535, decimal_places=65535)
    ipi = models.DecimalField(max_digits=65535, decimal_places=65535)

    class Meta:
        managed = False
        db_table = 'cabecalho_nota_compra_outras'


class CabecalhoNotaCompraServicos(models.Model):
    cd_nota = models.OneToOneField(CabecalhoNotaCompra, models.DO_NOTHING, db_column='cd_nota', primary_key=True)
    base_iss = models.DecimalField(max_digits=65535, decimal_places=65535)
    valor_iss = models.DecimalField(max_digits=65535, decimal_places=65535)
    valor_ir = models.DecimalField(max_digits=65535, decimal_places=65535)
    ir = models.DecimalField(max_digits=65535, decimal_places=65535)
    valor_csll = models.DecimalField(max_digits=65535, decimal_places=65535)
    csll = models.DecimalField(max_digits=65535, decimal_places=65535)
    valor_inss = models.DecimalField(max_digits=65535, decimal_places=65535)
    inss = models.DecimalField(max_digits=65535, decimal_places=65535)

    class Meta:
        managed = False
        db_table = 'cabecalho_nota_compra_servicos'


class CabecalhoNotaCompraTransferencia(models.Model):
    cd_nota = models.OneToOneField(CabecalhoNotaCompra, models.DO_NOTHING, db_column='cd_nota', primary_key=True)
    cd_transferencia_filial = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'cabecalho_nota_compra_transferencia'


class CabecalhoNotaCompraTransporte(models.Model):
    cd_nota = models.OneToOneField(CabecalhoNotaCompra, models.DO_NOTHING, db_column='cd_nota', primary_key=True)
    transportadora = models.ForeignKey('Fornecedores', models.DO_NOTHING, db_column='transportadora', blank=True, null=True)
    frete_por_conta = models.IntegerField()
    cfop = models.ForeignKey('Cfop', models.DO_NOTHING, db_column='cfop', blank=True, null=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535)
    seguro = models.DecimalField(max_digits=65535, decimal_places=65535)
    icms = models.DecimalField(max_digits=65535, decimal_places=65535)
    ipi = models.DecimalField(max_digits=65535, decimal_places=65535)
    placa = models.CharField(max_length=9, blank=True, null=True)
    uf_placa = models.CharField(max_length=8, blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    numero = models.CharField(max_length=60, blank=True, null=True)
    especie = models.CharField(max_length=60, blank=True, null=True)
    marca = models.CharField(max_length=60, blank=True, null=True)
    peso_liquido = models.DecimalField(max_digits=65535, decimal_places=65535)
    peso_bruto = models.DecimalField(max_digits=65535, decimal_places=65535)

    class Meta:
        managed = False
        db_table = 'cabecalho_nota_compra_transporte'


class CabecalhoOrcamento(models.Model):
    pedido = models.AutoField(primary_key=True)
    vendedor = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='vendedor', blank=True, null=True)
    cliente = models.ForeignKey('Clientes', models.DO_NOTHING, db_column='cliente', blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    validade = models.DateField(blank=True, null=True)
    transforma = models.CharField(max_length=1, blank=True, null=True)
    transportador = models.IntegerField(blank=True, null=True)
    nome = models.CharField(max_length=60, blank=True, null=True)
    observacao_impressa = models.ForeignKey('ObservacoesMovimentos', models.DO_NOTHING, db_column='observacao_impressa', blank=True, null=True)
    desconto = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    pedido_venda = models.IntegerField(blank=True, null=True)
    frete = models.DecimalField(max_digits=65535, decimal_places=65535)
    acrescimo = models.DecimalField(max_digits=65535, decimal_places=65535)
    outras = models.DecimalField(max_digits=65535, decimal_places=65535)
    seguro = models.DecimalField(max_digits=65535, decimal_places=65535)
    redespacho = models.ForeignKey('Fornecedores', models.DO_NOTHING, db_column='redespacho', blank=True, null=True)
    usuario_cancelamento = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario_cancelamento', blank=True, null=True)
    motivo_cancelamento = models.ForeignKey('Motivos', models.DO_NOTHING, db_column='motivo_cancelamento', blank=True, null=True)
    data_cancelamento = models.DateField(blank=True, null=True)
    hora_cancelamento = models.TimeField(blank=True, null=True)
    complemento_cancelamento = models.CharField(max_length=-1, blank=True, null=True)
    subst_parcela = models.BooleanField(blank=True, null=True)
    observacao = models.ForeignKey('ObservacoesMovimentos', models.DO_NOTHING, db_column='observacao', blank=True, null=True)
    hora_abertura = models.TimeField()
    especie = models.ForeignKey('Especies', models.DO_NOTHING, db_column='especie', blank=True, null=True)
    cupom = models.IntegerField()
    desconto_pontos = models.DecimalField(max_digits=15, decimal_places=2)

    class Meta:
        managed = False
        db_table = 'cabecalho_orcamento'


class CabecalhoOrcamentoComplementos(models.Model):
    orcamento = models.OneToOneField(CabecalhoOrcamento, models.DO_NOTHING, db_column='orcamento', primary_key=True)
    observacao = models.ForeignKey('ObservacoesMovimentos', models.DO_NOTHING, db_column='observacao', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_orcamento_complementos'


class CabecalhoOrcamentoFinanceiro(models.Model):
    orcamento = models.OneToOneField(CabecalhoOrcamento, models.DO_NOTHING, db_column='orcamento', primary_key=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    portador = models.ForeignKey(Bancos, models.DO_NOTHING, db_column='portador', blank=True, null=True)
    especie = models.ForeignKey('Especies', models.DO_NOTHING, db_column='especie', blank=True, null=True)
    cond_pgto = models.ForeignKey('Vencimentos', models.DO_NOTHING, db_column='cond_pgto', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_orcamento_financeiro'


class CabecalhoOrcamentoModelo(models.Model):
    pedido = models.AutoField(primary_key=True)
    descricao = models.CharField(max_length=200, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_orcamento_modelo'


class CabecalhoOrcamentoParcelas(models.Model):
    codigo = models.AutoField(primary_key=True)
    orcamento = models.ForeignKey(CabecalhoOrcamento, models.DO_NOTHING, db_column='orcamento', blank=True, null=True)
    parcela = models.IntegerField()
    vencimento = models.DateField()
    valor = models.DecimalField(max_digits=65535, decimal_places=65535)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    data_lancamento = models.DateField()
    hora_lancamento = models.TimeField()

    class Meta:
        managed = False
        db_table = 'cabecalho_orcamento_parcelas'


class CabecalhoOrcamentoPlacas(models.Model):
    ordem = models.OneToOneField(CabecalhoOrcamento, models.DO_NOTHING, db_column='ordem', primary_key=True)
    placa = models.CharField(max_length=8, blank=True, null=True)
    km = models.IntegerField(blank=True, null=True)
    prisma = models.CharField(max_length=3, blank=True, null=True)
    motorista = models.CharField(max_length=20, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_orcamento_placas'


class CabecalhoOrdemServico(models.Model):
    ordem = models.AutoField(primary_key=True)
    dataordem = models.DateField()
    cliente = models.ForeignKey('Clientes', models.DO_NOTHING, db_column='cliente', blank=True, null=True)
    vendedor = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='vendedor', blank=True, null=True)
    motor = models.CharField(max_length=100, blank=True, null=True)
    status = models.CharField(max_length=1)
    desconto = models.DecimalField(max_digits=12, decimal_places=2)
    multa = models.DecimalField(max_digits=12, decimal_places=2)
    acrescimo = models.DecimalField(max_digits=12, decimal_places=2)
    ncupom = models.IntegerField(blank=True, null=True)
    venda = models.CharField(max_length=1)
    transportadora = models.ForeignKey('Fornecedores', models.DO_NOTHING, db_column='transportadora', blank=True, null=True)
    frete = models.DecimalField(max_digits=12, decimal_places=3)
    especie = models.ForeignKey('Especies', models.DO_NOTHING, db_column='especie', blank=True, null=True)
    condicao_pgto = models.ForeignKey('Vencimentos', models.DO_NOTHING, db_column='condicao_pgto', blank=True, null=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador', blank=True, null=True)
    credito_usado = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    consignado = models.BooleanField(blank=True, null=True)
    observacao = models.ForeignKey('ObservacoesMovimentos', models.DO_NOTHING, db_column='observacao', blank=True, null=True)
    observacao_impressa = models.ForeignKey('ObservacoesMovimentos', models.DO_NOTHING, db_column='observacao_impressa', blank=True, null=True)
    redespacho = models.ForeignKey('Fornecedores', models.DO_NOTHING, db_column='redespacho', blank=True, null=True)
    outras_despesas = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    seguro = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    data_fechamento = models.DateField(blank=True, null=True)
    hora_abertura = models.TimeField()
    data_cancelamento = models.DateField(blank=True, null=True)
    hora_cancelamento = models.TimeField(blank=True, null=True)
    hora_fechamento = models.TimeField(blank=True, null=True)
    usuario_cancelamento = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario_cancelamento', blank=True, null=True)
    compl_cancelamento = models.CharField(max_length=50, blank=True, null=True)
    motivo_cancelamento = models.ForeignKey('Motivos', models.DO_NOTHING, db_column='motivo_cancelamento', blank=True, null=True)
    subst_parcela = models.BooleanField(blank=True, null=True)
    numero_serie_impressora = models.CharField(max_length=22, blank=True, null=True)
    comissao = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    indicador_presenca = models.IntegerField()
    usuarioautorizou = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuarioautorizou', blank=True, null=True)
    datahoraautorizacao = models.DateTimeField(blank=True, null=True)
    desconto_pontos = models.DecimalField(max_digits=15, decimal_places=2)
    fechando = models.BooleanField()
    cancelando = models.BooleanField()
    importando = models.BooleanField()
    numero_hd = models.CharField(max_length=8)
    impressao = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_ordem_servico'


class CabecalhoOrdemServicoAutomotivos(models.Model):
    placa = models.CharField(primary_key=True, max_length=8)
    nome = models.CharField(max_length=20, blank=True, null=True)
    chassi = models.CharField(max_length=20, blank=True, null=True)
    ano = models.IntegerField(blank=True, null=True)
    cor = models.CharField(max_length=20, blank=True, null=True)
    motor = models.CharField(max_length=20, blank=True, null=True)
    combustivel = models.CharField(max_length=20, blank=True, null=True)
    marca = models.ForeignKey('Marcas', models.DO_NOTHING, db_column='marca', blank=True, null=True)
    modelo = models.ForeignKey('Modelos', models.DO_NOTHING, db_column='modelo', blank=True, null=True)
    serie = models.CharField(max_length=20, blank=True, null=True)
    renavam = models.CharField(max_length=11, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_ordem_servico_automotivos'


class CabecalhoOrdemServicoAutorizados(models.Model):
    ordem = models.ForeignKey(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_ordem_servico_autorizados'


class CabecalhoOrdemServicoDadosCupom(models.Model):
    ordem = models.OneToOneField(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', primary_key=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_ordem_servico_dados_cupom'


class CabecalhoOrdemServicoDespesas(models.Model):
    codigo = models.AutoField(primary_key=True)
    ordem = models.ForeignKey(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', blank=True, null=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    preco_venda = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    preco_custo = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    reserva = models.ForeignKey(CabecalhoOrcamento, models.DO_NOTHING, db_column='reserva', blank=True, null=True)
    historico = models.ForeignKey('Historicos', models.DO_NOTHING, db_column='historico', blank=True, null=True)
    descricao = models.CharField(max_length=50, blank=True, null=True)
    item_remessa_venda_fora = models.ForeignKey('ItensRemessaVendaFora', models.DO_NOTHING, db_column='item_remessa_venda_fora', blank=True, null=True)
    item_nota_compra = models.ForeignKey('ItensNotaCompra', models.DO_NOTHING, db_column='item_nota_compra', blank=True, null=True)
    hora_lancamento = models.TimeField()
    utiliza_lote = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'cabecalho_ordem_servico_despesas'


class CabecalhoOrdemServicoErros(models.Model):
    codigo = models.AutoField(primary_key=True)
    ordem = models.IntegerField(blank=True, null=True)
    descricao = models.CharField(max_length=-1, blank=True, null=True)
    operador = models.IntegerField()
    data_lancamento = models.DateField()
    hora_lancamento = models.TimeField()

    class Meta:
        managed = False
        db_table = 'cabecalho_ordem_servico_erros'


class CabecalhoOrdemServicoExportacao(models.Model):
    ordem = models.OneToOneField(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', primary_key=True)
    cep = models.ForeignKey('Cidades', models.DO_NOTHING, db_column='cep')
    local_despacho = models.CharField(max_length=60)

    class Meta:
        managed = False
        db_table = 'cabecalho_ordem_servico_exportacao'


class CabecalhoOrdemServicoExportados(models.Model):
    ordem = models.OneToOneField(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', primary_key=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_ordem_servico_exportados'


class CabecalhoOrdemServicoLiberacaoFaturamento(models.Model):
    codigo = models.AutoField(primary_key=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    datahora = models.DateTimeField()
    usuarioautorizou = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuarioautorizou')
    ordem = models.ForeignKey(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem')
    cliente = models.ForeignKey('Clientes', models.DO_NOTHING, db_column='cliente')

    class Meta:
        managed = False
        db_table = 'cabecalho_ordem_servico_liberacao_faturamento'


class CabecalhoOrdemServicoLimite(models.Model):
    pedido = models.OneToOneField(CabecalhoOrdemServico, models.DO_NOTHING, db_column='pedido', primary_key=True)
    valor_autorizado = models.DecimalField(max_digits=65535, decimal_places=65535)
    limite = models.DecimalField(max_digits=65535, decimal_places=65535)
    saldo_devedor = models.DecimalField(max_digits=65535, decimal_places=65535)
    credito = models.DecimalField(max_digits=65535, decimal_places=65535)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    data_lancamento = models.DateField()
    hora_lancamento = models.TimeField()

    class Meta:
        managed = False
        db_table = 'cabecalho_ordem_servico_limite'


class CabecalhoOrdemServicoOs(models.Model):
    ordem = models.OneToOneField(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', primary_key=True)
    status = models.IntegerField(blank=True, null=True)
    numeroserie = models.CharField(max_length=20, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_ordem_servico_os'


class CabecalhoOrdemServicoOsExcluido(models.Model):
    ordem = models.OneToOneField(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', primary_key=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_ordem_servico_os_excluido'


class CabecalhoOrdemServicoPalm(models.Model):
    pedido = models.OneToOneField('PalmPedidos', models.DO_NOTHING, db_column='pedido', primary_key=True)
    ordem = models.ForeignKey(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem')
    operador = models.IntegerField(blank=True, null=True)
    data = models.DateTimeField(blank=True, null=True)
    portador = models.IntegerField(blank=True, null=True)
    especie = models.IntegerField(blank=True, null=True)
    condicao_pgto = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_ordem_servico_palm'
        unique_together = (('pedido', 'ordem'),)


class CabecalhoOrdemServicoParcelasFinanceira(models.Model):
    codigo = models.AutoField(primary_key=True)
    ordem = models.ForeignKey(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', blank=True, null=True)
    parcela = models.IntegerField()
    vencimento = models.DateField()
    valor = models.DecimalField(max_digits=65535, decimal_places=65535)
    operador = models.IntegerField()
    data_lancamento = models.DateField()
    hora_lancamento = models.TimeField()

    class Meta:
        managed = False
        db_table = 'cabecalho_ordem_servico_parcelas_financeira'


class CabecalhoOrdemServicoPlacas(models.Model):
    ordem = models.OneToOneField(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', primary_key=True)
    placa = models.ForeignKey(CabecalhoOrdemServicoAutomotivos, models.DO_NOTHING, db_column='placa', blank=True, null=True)
    km = models.IntegerField(blank=True, null=True)
    prisma = models.CharField(max_length=3, blank=True, null=True)
    motorista = models.CharField(max_length=20, blank=True, null=True)
    md5 = models.CharField(max_length=32)

    class Meta:
        managed = False
        db_table = 'cabecalho_ordem_servico_placas'


class CabecalhoOrdemServicoPreVenda(models.Model):
    ordem = models.OneToOneField(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', primary_key=True)
    cancelado = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'cabecalho_ordem_servico_pre_venda'


class CabecalhoOrdemServicoProducao(models.Model):
    pedido_venda = models.OneToOneField(CabecalhoOrdemServico, models.DO_NOTHING, db_column='pedido_venda', primary_key=True)
    ordem_producao = models.ForeignKey('OrdemProducao', models.DO_NOTHING, db_column='ordem_producao', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_ordem_servico_producao'


class CabecalhoOrdemServicoProporcionaisContingencia(models.Model):
    ordem = models.OneToOneField(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', primary_key=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    datahora = models.DateTimeField()
    desconto = models.DecimalField(max_digits=12, decimal_places=2)
    desconto_pontos = models.DecimalField(max_digits=12, decimal_places=2)
    acrescimo = models.DecimalField(max_digits=12, decimal_places=2)
    frete = models.DecimalField(max_digits=12, decimal_places=2)
    seguro = models.DecimalField(max_digits=12, decimal_places=2)
    outras_despesas = models.DecimalField(max_digits=12, decimal_places=2)

    class Meta:
        managed = False
        db_table = 'cabecalho_ordem_servico_proporcionais_contingencia'


class CabecalhoOrdemServicoReceituario(models.Model):
    ordem = models.OneToOneField(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', primary_key=True)
    dependente = models.ForeignKey('ClientesDependentes', models.DO_NOTHING, db_column='dependente', blank=True, null=True)
    longe_od_esf = models.CharField(max_length=12, blank=True, null=True)
    longe_od_cil = models.CharField(max_length=12, blank=True, null=True)
    longe_od_eixo = models.CharField(max_length=12, blank=True, null=True)
    longe_od_adicao = models.CharField(max_length=12, blank=True, null=True)
    longe_od_dp = models.CharField(max_length=12, blank=True, null=True)
    perto_od_esf = models.CharField(max_length=12, blank=True, null=True)
    perto_od_cil = models.CharField(max_length=12, blank=True, null=True)
    perto_od_eixo = models.CharField(max_length=12, blank=True, null=True)
    perto_od_dp = models.CharField(max_length=12, blank=True, null=True)
    longe_oe_esf = models.CharField(max_length=12, blank=True, null=True)
    longe_oe_cil = models.CharField(max_length=12, blank=True, null=True)
    longe_oe_eixo = models.CharField(max_length=12, blank=True, null=True)
    longe_oe_adicao = models.CharField(max_length=12, blank=True, null=True)
    longe_oe_dp = models.CharField(max_length=12, blank=True, null=True)
    perto_oe_esf = models.CharField(max_length=12, blank=True, null=True)
    perto_oe_cil = models.CharField(max_length=12, blank=True, null=True)
    perto_oe_eixo = models.CharField(max_length=12, blank=True, null=True)
    perto_oe_dp = models.CharField(max_length=12, blank=True, null=True)
    data_ent = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)
    tipo_lente = models.CharField(max_length=12, blank=True, null=True)
    cor = models.CharField(max_length=15, blank=True, null=True)
    medico = models.CharField(max_length=50, blank=True, null=True)
    perto_od_adicao = models.CharField(max_length=12, blank=True, null=True)
    perto_oe_adicao = models.CharField(max_length=12, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_ordem_servico_receituario'


class CabecalhoOrdemServicoRestaurante(models.Model):
    ordem = models.OneToOneField(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', primary_key=True)
    conta_cliente = models.IntegerField(blank=True, null=True)
    mesa = models.IntegerField(blank=True, null=True)
    status = models.IntegerField(blank=True, null=True)
    md5 = models.CharField(max_length=32)

    class Meta:
        managed = False
        db_table = 'cabecalho_ordem_servico_restaurante'


class CabecalhoOrdemServicoRestauranteExcluido(models.Model):
    ordem = models.OneToOneField(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', primary_key=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_ordem_servico_restaurante_excluido'


class CabecalhoOrdemServicoTotais(models.Model):
    ordem = models.OneToOneField(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', primary_key=True)
    total_pedido = models.DecimalField(max_digits=65535, decimal_places=65535)
    total_pedido_bruto = models.DecimalField(max_digits=65535, decimal_places=65535)
    md5 = models.CharField(max_length=-1)

    class Meta:
        managed = False
        db_table = 'cabecalho_ordem_servico_totais'


class CabecalhoOrdemServicoTransporte(models.Model):
    ordem = models.OneToOneField(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', primary_key=True)
    frete_conta = models.IntegerField()
    peso_liquido = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    peso_bruto = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    volume = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_ordem_servico_transporte'


class CabecalhoPedidoCompra(models.Model):
    pedido = models.AutoField(primary_key=True)
    fornecedor = models.ForeignKey('Fornecedores', models.DO_NOTHING, db_column='fornecedor', blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    fecha = models.CharField(max_length=1, blank=True, null=True)
    observacoes = models.ForeignKey('ObservacoesMovimentos', models.DO_NOTHING, db_column='observacoes', blank=True, null=True)
    converteu = models.BooleanField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_pedido_compra'


class CabecalhoRemessaVendaFora(models.Model):
    pedido = models.AutoField(primary_key=True)
    data = models.DateField()
    cliente = models.ForeignKey('Clientes', models.DO_NOTHING, db_column='cliente', blank=True, null=True)
    transportadora = models.ForeignKey('Fornecedores', models.DO_NOTHING, db_column='transportadora', blank=True, null=True)
    data_fechamento = models.DateField(blank=True, null=True)
    hora_fechamento = models.TimeField(blank=True, null=True)
    data_lancamento = models.DateField()
    hora_lancamento = models.TimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    vendedor = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='vendedor', blank=True, null=True)
    enviou_palm = models.BooleanField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_remessa_venda_fora'


class CabecalhoRemessaVendaForaCancelados(models.Model):
    pedido = models.OneToOneField(CabecalhoRemessaVendaFora, models.DO_NOTHING, db_column='pedido', primary_key=True)
    data_cancelamento = models.DateField()
    hora_cancelamento = models.TimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'cabecalho_remessa_venda_fora_cancelados'


class CabecalhoRemessaVendaForaOutrasDespesas(models.Model):
    pedido = models.OneToOneField(CabecalhoRemessaVendaFora, models.DO_NOTHING, db_column='pedido', primary_key=True)
    frete = models.DecimalField(max_digits=65535, decimal_places=65535)
    acrescimo = models.DecimalField(max_digits=65535, decimal_places=65535)
    desconto = models.DecimalField(max_digits=65535, decimal_places=65535)
    seguro = models.DecimalField(max_digits=65535, decimal_places=65535)
    outras = models.DecimalField(max_digits=65535, decimal_places=65535)

    class Meta:
        managed = False
        db_table = 'cabecalho_remessa_venda_fora_outras_despesas'


class CabecalhoRetornoVendaFora(models.Model):
    pedido = models.AutoField(primary_key=True)
    data = models.DateField()
    cliente = models.ForeignKey('Clientes', models.DO_NOTHING, db_column='cliente', blank=True, null=True)
    transportadora = models.ForeignKey('Fornecedores', models.DO_NOTHING, db_column='transportadora', blank=True, null=True)
    data_fechamento = models.DateField(blank=True, null=True)
    hora_fechamento = models.TimeField(blank=True, null=True)
    data_lancamento = models.DateField()
    hora_lancamento = models.TimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    vendedor = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='vendedor', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_retorno_venda_fora'


class CabecalhoRetornoVendaForaCancelados(models.Model):
    pedido = models.OneToOneField(CabecalhoRetornoVendaFora, models.DO_NOTHING, db_column='pedido', primary_key=True)
    data_cancelamento = models.DateField()
    hora_cancelamento = models.TimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'cabecalho_retorno_venda_fora_cancelados'


class CabecalhoRetornoVendaForaOutrasDespesas(models.Model):
    pedido = models.OneToOneField(CabecalhoRetornoVendaFora, models.DO_NOTHING, db_column='pedido', primary_key=True)
    frete = models.DecimalField(max_digits=65535, decimal_places=65535)
    acrescimo = models.DecimalField(max_digits=65535, decimal_places=65535)
    desconto = models.DecimalField(max_digits=65535, decimal_places=65535)
    seguro = models.DecimalField(max_digits=65535, decimal_places=65535)
    outras = models.DecimalField(max_digits=65535, decimal_places=65535)

    class Meta:
        managed = False
        db_table = 'cabecalho_retorno_venda_fora_outras_despesas'


class CabecalhoRomaneio(models.Model):
    romaneio = models.AutoField(primary_key=True)
    data = models.DateField()
    placa = models.ForeignKey('Veiculos', models.DO_NOTHING, db_column='placa')
    carga_maxima = models.DecimalField(max_digits=65535, decimal_places=65535)
    transportadora = models.IntegerField()
    motorista = models.CharField(max_length=100, blank=True, null=True)
    nome = models.CharField(max_length=100)
    data_hora = models.DateTimeField()
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')

    class Meta:
        managed = False
        db_table = 'cabecalho_romaneio'


class CabecalhoRomaneioCanceladas(models.Model):
    romaneio = models.OneToOneField(CabecalhoRomaneio, models.DO_NOTHING, db_column='romaneio', primary_key=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    data_hora = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'cabecalho_romaneio_canceladas'


class CabecalhoRomaneioFechados(models.Model):
    romaneio = models.OneToOneField(CabecalhoRomaneio, models.DO_NOTHING, db_column='romaneio', primary_key=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    data_hora = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'cabecalho_romaneio_fechados'


class CabecalhoRomaneioObservacoes(models.Model):
    romaneio = models.OneToOneField(CabecalhoRomaneio, models.DO_NOTHING, db_column='romaneio', primary_key=True)
    observacao = models.TextField(blank=True, null=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    data_hora = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'cabecalho_romaneio_observacoes'


class CabecalhoSeparacaoEtiquetas(models.Model):
    codigo = models.AutoField(primary_key=True)
    data_operacao = models.DateTimeField()
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')

    class Meta:
        managed = False
        db_table = 'cabecalho_separacao_etiquetas'


class CabecalhoServicosPlacas(models.Model):
    ordem = models.OneToOneField('Servicos', models.DO_NOTHING, db_column='ordem', primary_key=True)
    placa = models.CharField(max_length=8, blank=True, null=True)
    km = models.IntegerField(blank=True, null=True)
    prisma = models.CharField(max_length=100, blank=True, null=True)
    motorista = models.CharField(max_length=-1, blank=True, null=True)
    md5 = models.CharField(max_length=32)

    class Meta:
        managed = False
        db_table = 'cabecalho_servicos_placas'


class CabecalhoTabelasPromocionais(models.Model):
    codigo = models.AutoField(primary_key=True)
    descricao = models.CharField(max_length=50, blank=True, null=True)
    validade = models.DateField(blank=True, null=True)
    repassou = models.BooleanField()
    validade_inicial = models.DateField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cabecalho_tabelas_promocionais'


class CabecalhosDespesas(models.Model):
    codigo = models.AutoField(primary_key=True)
    devolucao = models.ForeignKey(CabecalhoDevolucao, models.DO_NOTHING, db_column='devolucao', blank=True, null=True)
    outras = models.DecimalField(max_digits=65535, decimal_places=65535)
    seguro = models.DecimalField(max_digits=65535, decimal_places=65535)
    acrescimo = models.DecimalField(max_digits=65535, decimal_places=65535)
    desconto = models.DecimalField(max_digits=65535, decimal_places=65535)
    frete = models.DecimalField(max_digits=65535, decimal_places=65535)
    tipo_frete = models.IntegerField()
    icms_frete = models.DecimalField(max_digits=65535, decimal_places=65535)

    class Meta:
        managed = False
        db_table = 'cabecalhos_despesas'


class Caixa(models.Model):
    data = models.DateField(blank=True, null=True)
    codhis = models.ForeignKey('Historicos', models.DO_NOTHING, db_column='codhis', blank=True, null=True)
    codconta = models.IntegerField(blank=True, null=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)
    operador = models.IntegerField(blank=True, null=True)
    origem = models.IntegerField(blank=True, null=True)
    item_origem = models.IntegerField(blank=True, null=True)
    abre_fecha_origem = models.ForeignKey(AbreFechaCaixa, models.DO_NOTHING, db_column='abre_fecha_origem', blank=True, null=True)
    codigo = models.AutoField(primary_key=True)
    complemento = models.CharField(max_length=200, blank=True, null=True)
    parcela = models.ForeignKey('Parcelas', models.DO_NOTHING, db_column='parcela', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'caixa'


class CaixaConfiguracao(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    nome = models.CharField(max_length=100)
    abertura = models.TimeField()
    fechamento = models.TimeField()

    class Meta:
        managed = False
        db_table = 'caixa_configuracao'


class CaixaEspecies(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    caixa = models.ForeignKey(Caixa, models.DO_NOTHING, db_column='caixa')
    especie = models.ForeignKey('Especies', models.DO_NOTHING, db_column='especie')
    valor = models.DecimalField(max_digits=15, decimal_places=2)
    valor_pago = models.DecimalField(max_digits=15, decimal_places=2)

    class Meta:
        managed = False
        db_table = 'caixa_especies'


class CamposContratos(models.Model):
    codigo = models.AutoField(primary_key=True)
    nome = models.CharField(max_length=-1, blank=True, null=True)
    imprime = models.BooleanField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'campos_contratos'


class Cargos(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nome = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cargos'


class CartaCorrecao(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_nota = models.ForeignKey('NotasEletronicas', models.DO_NOTHING, db_column='cd_nota')
    id = models.CharField(max_length=54)
    status = models.IntegerField()
    mensagem = models.CharField(max_length=5000, blank=True, null=True)
    observacao = models.ForeignKey('Observacoes', models.DO_NOTHING, db_column='observacao', blank=True, null=True)
    datahoracartacorrecao = models.CharField(max_length=50, blank=True, null=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    marca = models.CharField(max_length=1, blank=True, null=True)
    ambiente = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'carta_correcao'


class CartaCorrecaoEmail(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_carta = models.ForeignKey(CartaCorrecao, models.DO_NOTHING, db_column='cd_carta')
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    texto = models.CharField(max_length=5000)

    class Meta:
        managed = False
        db_table = 'carta_correcao_email'


class CartaCorrecaoXml(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_carta = models.ForeignKey(CartaCorrecao, models.DO_NOTHING, db_column='cd_carta')
    xml = models.TextField()
    impressao = models.TextField(blank=True, null=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'carta_correcao_xml'


class Centrocustos(models.Model):
    codigo = models.IntegerField(primary_key=True)
    descricao = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'centrocustos'


class Cest(models.Model):
    codigo = models.AutoField(primary_key=True)
    ncm = models.CharField(max_length=8, blank=True, null=True)
    cest = models.CharField(max_length=7, blank=True, null=True)
    descricao = models.CharField(max_length=-1, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cest'


class Cfop(models.Model):
    cfop = models.CharField(primary_key=True, max_length=20)
    descricao = models.CharField(max_length=50)
    obs_nota = models.CharField(max_length=100, blank=True, null=True)
    obs_geral = models.CharField(max_length=100, blank=True, null=True)
    nota_tributada = models.CharField(max_length=1)
    movimenta_estoque = models.CharField(max_length=1)
    acumulador1 = models.IntegerField(blank=True, null=True)
    acumulador2 = models.IntegerField(blank=True, null=True)
    atualizacao = models.BooleanField(blank=True, null=True)
    bonificado = models.CharField(max_length=1)
    ativo = models.BooleanField()
    bonificado_destaca_icms = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'cfop'


class CfopCsts(models.Model):
    cfop = models.IntegerField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'cfop_csts'


class CfopCstsPermitidos(models.Model):
    cfop = models.ForeignKey(CfopCsts, models.DO_NOTHING, db_column='cfop')
    cst = models.CharField(max_length=-1)

    class Meta:
        managed = False
        db_table = 'cfop_csts_permitidos'
        unique_together = (('cfop', 'cst'),)


class CfopDevolucao(models.Model):
    cfop = models.CharField(primary_key=True, max_length=4)

    class Meta:
        managed = False
        db_table = 'cfop_devolucao'


class CfopOperacoes(models.Model):
    codigo = models.AutoField(primary_key=True)
    cfop_venda = models.CharField(max_length=4, blank=True, null=True)
    cfop_compra = models.CharField(max_length=4, blank=True, null=True)
    operacaoentrada = models.IntegerField()
    operacaosaida = models.IntegerField()
    operacaocomst = models.BooleanField()
    cstentrada = models.CharField(max_length=4)
    contribuinte = models.IntegerField()
    icmsretidoanteriormente = models.IntegerField()
    tipomercadoria = models.CharField(max_length=2, blank=True, null=True)
    destinomercadoria = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cfop_operacoes'


class CheckNumeroSerie(models.Model):
    codigo = models.OneToOneField('TerminaisDados', models.DO_NOTHING, db_column='codigo', primary_key=True)
    numero_serie = models.CharField(max_length=-1, blank=True, null=True)
    exportou = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'check_numero_serie'


class ChequesAlteracao(models.Model):
    codigo = models.AutoField(primary_key=True)
    codigo_cheque = models.ForeignKey('Chequespre', models.DO_NOTHING, db_column='codigo_cheque', blank=True, null=True)
    data_anterior = models.DateField(blank=True, null=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    fatura = models.ForeignKey('Faturas', models.DO_NOTHING, db_column='fatura', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cheques_alteracao'


class ChequesEmitidos(models.Model):
    codigo = models.AutoField(primary_key=True)
    ctabancaria = models.ForeignKey('Ctabancarias', models.DO_NOTHING, db_column='ctabancaria', blank=True, null=True)
    cheque = models.IntegerField(blank=True, null=True)
    data_emissao = models.DateField(blank=True, null=True)
    hora_emissao = models.TimeField(blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)
    cd_apagar = models.ForeignKey(Apagar, models.DO_NOTHING, db_column='cd_apagar', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cheques_emitidos'


class Chequespre(models.Model):
    pedidonro = models.IntegerField(blank=True, null=True)
    clientenro = models.IntegerField(blank=True, null=True)
    parcelanro = models.IntegerField(blank=True, null=True)
    chequenro = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    banconro = models.IntegerField(blank=True, null=True)
    contanro = models.CharField(max_length=15, blank=True, null=True)
    valor = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    emitente = models.CharField(max_length=50, blank=True, null=True)
    datapre = models.DateField(blank=True, null=True)
    compensado = models.DateField(blank=True, null=True)
    destino = models.CharField(max_length=50, blank=True, null=True)
    devolvido = models.CharField(max_length=1, blank=True, null=True)
    codigo = models.AutoField(primary_key=True)
    observacao = models.IntegerField(blank=True, null=True)
    agencia = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'chequespre'


class Cidades(models.Model):
    cep = models.CharField(primary_key=True, max_length=9)
    cidade = models.CharField(max_length=50, blank=True, null=True)
    uf = models.ForeignKey('Estados', models.DO_NOTHING, db_column='uf', blank=True, null=True)
    codigo_cidade = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cidades'


class ClassFiscal(models.Model):
    codigo = models.AutoField(primary_key=True)
    nome = models.CharField(max_length=50)

    class Meta:
        managed = False
        db_table = 'class_fiscal'


class ClassFiscalProd(models.Model):
    codigo = models.AutoField(primary_key=True)
    uf = models.ForeignKey('Estados', models.DO_NOTHING, db_column='uf', blank=True, null=True)
    class_fiscal = models.ForeignKey(ClassFiscal, models.DO_NOTHING, db_column='class_fiscal', blank=True, null=True)
    obs_lei = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'class_fiscal_prod'


class ClasseProdutos(models.Model):
    codigo = models.AutoField(primary_key=True)
    nome = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'classe_produtos'


class ClassesProfissionais(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nome = models.CharField(max_length=50, blank=True, null=True)
    codigo_financeira = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'classes_profissionais'


class ClienteLogAlteracoes(models.Model):
    codigo = models.AutoField(primary_key=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    data = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)
    campo = models.CharField(max_length=-1, blank=True, null=True)
    antigo = models.CharField(max_length=-1, blank=True, null=True)
    novo = models.CharField(max_length=-1, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cliente_log_alteracoes'


class Clientes(models.Model):
    codigo = models.AutoField(primary_key=True)
    grupo = models.ForeignKey('GrupoClientes', models.DO_NOTHING, db_column='grupo', blank=True, null=True)
    atividade = models.ForeignKey(Atividades, models.DO_NOTHING, db_column='atividade', blank=True, null=True)
    nome = models.CharField(max_length=50, blank=True, null=True)
    tipo = models.IntegerField(blank=True, null=True)
    endereco = models.CharField(max_length=50, blank=True, null=True)
    bairro = models.CharField(max_length=50, blank=True, null=True)
    cep = models.ForeignKey(Cidades, models.DO_NOTHING, db_column='cep', blank=True, null=True)
    fone = models.CharField(max_length=14, blank=True, null=True)
    datacad = models.DateField(blank=True, null=True)
    vendedorpadrao = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='vendedorpadrao', blank=True, null=True)
    fax = models.CharField(max_length=14, blank=True, null=True)
    celular = models.CharField(max_length=14, blank=True, null=True)
    email = models.CharField(max_length=500, blank=True, null=True)
    url = models.CharField(max_length=200, blank=True, null=True)
    regiao = models.ForeignKey('Regioes', models.DO_NOTHING, db_column='regiao', blank=True, null=True)
    desconto = models.DecimalField(max_digits=6, decimal_places=2, blank=True, null=True)
    estadocivil = models.IntegerField(blank=True, null=True)
    cpf_cnpj = models.CharField(max_length=18)
    inscricaoest = models.CharField(max_length=20)
    fantasia = models.CharField(max_length=50, blank=True, null=True)
    contato = models.CharField(max_length=50, blank=True, null=True)
    fone2 = models.CharField(max_length=14, blank=True, null=True)
    icms = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    complemento = models.CharField(max_length=255, blank=True, null=True)
    tabela_preco = models.ForeignKey(CabecaPreco, models.DO_NOTHING, db_column='tabela_preco', blank=True, null=True)
    condicao_pgto = models.ForeignKey('Vencimentos', models.DO_NOTHING, db_column='condicao_pgto', blank=True, null=True)
    especie = models.ForeignKey('Especies', models.DO_NOTHING, db_column='especie', blank=True, null=True)
    numero = models.CharField(max_length=10, blank=True, null=True)
    codobservacao2 = models.OneToOneField('Observacoes', models.DO_NOTHING, db_column='codobservacao2', blank=True, null=True)
    codobservacao = models.ForeignKey('Observacoes', models.DO_NOTHING, db_column='codobservacao', blank=True, null=True)
    imprime_ncm = models.BooleanField(blank=True, null=True)
    status = models.CharField(max_length=1, blank=True, null=True)
    recebe_lista = models.BooleanField(blank=True, null=True)
    dia_vencimento = models.IntegerField(blank=True, null=True)
    ramal = models.IntegerField(blank=True, null=True)
    prospectador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='prospectador', blank=True, null=True)
    marca = models.CharField(max_length=1, blank=True, null=True)
    rota = models.ForeignKey('Rotas', models.DO_NOTHING, db_column='rota', blank=True, null=True)
    permite_duplicar_cpf_cnpj = models.BooleanField(blank=True, null=True)
    avalista = models.ForeignKey('self', models.DO_NOTHING, db_column='avalista', blank=True, null=True)
    portador = models.ForeignKey(Bancos, models.DO_NOTHING, db_column='portador', blank=True, null=True)
    codigo_pais = models.ForeignKey('Paises', models.DO_NOTHING, db_column='codigo_pais', blank=True, null=True)
    fone_tipo = models.IntegerField()
    fone2_tipo = models.IntegerField()
    fax_tipo = models.IntegerField()
    celular_tipo = models.IntegerField()
    nascionalidade = models.IntegerField()
    sexo = models.IntegerField()
    dias_atualizar = models.IntegerField(blank=True, null=True)
    reside_desde = models.DateField(blank=True, null=True)
    codigo_checkexpress = models.IntegerField(blank=True, null=True)
    email2 = models.CharField(max_length=500, blank=True, null=True)
    envia_palm = models.BooleanField(blank=True, null=True)
    cobrarnfe = models.BooleanField()
    produtor_rural = models.BooleanField()
    simples = models.BooleanField()
    ficha_assinada = models.BooleanField()
    beneficio = models.IntegerField()
    email_envio_xml = models.IntegerField()
    id_estrangeiro = models.CharField(max_length=20, blank=True, null=True)
    beneficio_novo = models.IntegerField()
    ipi_cst = models.ForeignKey('CodigosFiscaisIpi', models.DO_NOTHING, db_column='ipi_cst')
    ipi_aliquota = models.DecimalField(max_digits=15, decimal_places=2)
    senha_md5 = models.CharField(max_length=-1, blank=True, null=True)
    destino_mercadoria = models.IntegerField()
    cfop_industrializacao = models.IntegerField(blank=True, null=True)
    calcula_icms_st = models.BooleanField()
    inscricaomunicipal = models.CharField(max_length=20)
    iss = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    retemiss = models.IntegerField(blank=True, null=True)
    desonerado = models.ForeignKey('Desonerado', models.DO_NOTHING, db_column='desonerado', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'clientes'


class ClientesAlteracoesSped(models.Model):
    codigo = models.AutoField(primary_key=True)
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente', blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    campo = models.IntegerField(blank=True, null=True)
    antigo = models.CharField(max_length=-1, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'clientes_alteracoes_sped'


class ClientesAnexos(models.Model):
    codigo = models.AutoField(primary_key=True)
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente', blank=True, null=True)
    nome = models.CharField(max_length=255, blank=True, null=True)
    anexo = models.BinaryField()
    fornecedor = models.ForeignKey('Fornecedores', models.DO_NOTHING, db_column='fornecedor', blank=True, null=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'clientes_anexos'


class ClientesAutorizados(models.Model):
    codigo = models.AutoField(primary_key=True)
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente', blank=True, null=True)
    autorizado = models.CharField(max_length=255, blank=True, null=True)
    marca = models.CharField(max_length=1, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'clientes_autorizados'


class ClientesDadosAlvara(models.Model):
    cliente = models.OneToOneField(Clientes, models.DO_NOTHING, db_column='cliente', primary_key=True)
    alvara = models.CharField(max_length=50)
    data_alvara = models.DateField(blank=True, null=True)
    certificado = models.CharField(max_length=50)
    validade_certificado = models.DateField(blank=True, null=True)
    responsavel = models.CharField(max_length=50)
    cr_responsavel = models.CharField(max_length=50)

    class Meta:
        managed = False
        db_table = 'clientes_dados_alvara'


class ClientesDependentes(models.Model):
    codigo = models.AutoField(primary_key=True)
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente', blank=True, null=True)
    dependente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='dependente', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'clientes_dependentes'


class ClientesFornecedoresRelacaoYandeh(models.Model):
    codigo = models.AutoField(primary_key=True)
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente', blank=True, null=True)
    fornecedor = models.ForeignKey('Fornecedores', models.DO_NOTHING, db_column='fornecedor', blank=True, null=True)
    id_yandeh = models.CharField(max_length=20, blank=True, null=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador', blank=True, null=True)
    data = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'clientes_fornecedores_relacao_yandeh'


class ClientesMensagemNf(models.Model):
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente', blank=True, null=True)
    codigo_obs = models.ForeignKey('Mensagemnf', models.DO_NOTHING, db_column='codigo_obs')

    class Meta:
        managed = False
        db_table = 'clientes_mensagem_nf'


class ClientesReferencias(models.Model):
    codigo = models.AutoField(primary_key=True)
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente', blank=True, null=True)
    nome = models.CharField(max_length=80, blank=True, null=True)
    relacionamento = models.CharField(max_length=80, blank=True, null=True)
    telefone = models.CharField(max_length=14, blank=True, null=True)
    ramal = models.CharField(max_length=5, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'clientes_referencias'


class ClientesRg(models.Model):
    cliente = models.OneToOneField(Clientes, models.DO_NOTHING, db_column='cliente', primary_key=True)
    orgao_emissor = models.IntegerField(blank=True, null=True)
    estado = models.ForeignKey('Estados', models.DO_NOTHING, db_column='estado', blank=True, null=True)
    data_emissao = models.DateField()

    class Meta:
        managed = False
        db_table = 'clientes_rg'


class ClientesSaldoDevedor(models.Model):
    cliente = models.OneToOneField(Clientes, models.DO_NOTHING, db_column='cliente', primary_key=True)
    saldo = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    filial = models.CharField(max_length=18, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'clientes_saldo_devedor'


class ClientesTransportadoras(models.Model):
    cliente = models.OneToOneField(Clientes, models.DO_NOTHING, db_column='cliente', primary_key=True)
    transportadora = models.ForeignKey('Fornecedores', models.DO_NOTHING, db_column='transportadora', blank=True, null=True)
    redespacho = models.ForeignKey('Fornecedores', models.DO_NOTHING, db_column='redespacho', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'clientes_transportadoras'


class CnaeProdutos(models.Model):
    codigo = models.AutoField(primary_key=True)
    nome = models.CharField(max_length=50, blank=True, null=True)
    codigotributacaomunicipio = models.CharField(max_length=30, blank=True, null=True)
    itemlistaservico = models.CharField(max_length=15, blank=True, null=True)
    tributamunicipioprestador = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'cnae_produtos'


class CodigosFiscais(models.Model):
    cst = models.CharField(primary_key=True, max_length=4)
    descricao = models.CharField(max_length=1000, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'codigos_fiscais'


class CodigosFiscaisCofins(models.Model):
    cst = models.CharField(primary_key=True, max_length=2)
    descricao = models.CharField(max_length=1000, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'codigos_fiscais_cofins'


class CodigosFiscaisIpi(models.Model):
    cst = models.CharField(primary_key=True, max_length=2)
    descricao = models.CharField(max_length=1000, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'codigos_fiscais_ipi'


class CodigosFiscaisIpiEnquadramento(models.Model):
    enquadramento = models.CharField(primary_key=True, max_length=3)
    descricao = models.CharField(max_length=1000, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'codigos_fiscais_ipi_enquadramento'


class CodigosFiscaisPis(models.Model):
    cst = models.CharField(primary_key=True, max_length=2)
    descricao = models.CharField(max_length=1000, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'codigos_fiscais_pis'


class ComandaMovimentos(models.Model):
    codigo = models.IntegerField(primary_key=True)
    venda = models.ForeignKey(CabecalhoOrdemServico, models.DO_NOTHING, db_column='venda', blank=True, null=True)
    orcamento = models.ForeignKey(CabecalhoOrcamento, models.DO_NOTHING, db_column='orcamento', blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    data = models.DateField()
    hora = models.TimeField()

    class Meta:
        managed = False
        db_table = 'comanda_movimentos'


class ComandaPedido(models.Model):
    ordem = models.OneToOneField(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', primary_key=True)
    comanda = models.IntegerField()
    data = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'comanda_pedido'
        unique_together = (('ordem', 'comanda'),)


class ComposicaoMovimento(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    item_ordem_servico = models.ForeignKey('ItensOrdemServico', models.DO_NOTHING, db_column='item_ordem_servico', blank=True, null=True)
    item_transferencia = models.ForeignKey('ItensTransferencia', models.DO_NOTHING, db_column='item_transferencia', blank=True, null=True)
    item_transferencia_entrada = models.ForeignKey('ItensTransferenciaEntrada', models.DO_NOTHING, db_column='item_transferencia_entrada', blank=True, null=True)
    item_remessa_conserto = models.ForeignKey('ItensRemessa', models.DO_NOTHING, db_column='item_remessa_conserto', blank=True, null=True)
    item_devolucao_fornecedor = models.ForeignKey('ItensRemessa', models.DO_NOTHING, db_column='item_devolucao_fornecedor', blank=True, null=True)
    item_compra = models.ForeignKey('ItensNotaCompra', models.DO_NOTHING, db_column='item_compra', blank=True, null=True)
    item_avulsa = models.ForeignKey('ItensNotaAvulsa', models.DO_NOTHING, db_column='item_avulsa', blank=True, null=True)
    item_devolucao = models.ForeignKey('ItensDevolucao', models.DO_NOTHING, db_column='item_devolucao', blank=True, null=True)
    item_futura = models.ForeignKey('ItensEntrega', models.DO_NOTHING, db_column='item_futura', blank=True, null=True)
    item_formula = models.ForeignKey('ItensFormula', models.DO_NOTHING, db_column='item_formula', blank=True, null=True)
    item_perda = models.ForeignKey('ItensPerda', models.DO_NOTHING, db_column='item_perda', blank=True, null=True)
    item_producao = models.ForeignKey('ItensProducao', models.DO_NOTHING, db_column='item_producao', blank=True, null=True)
    item_entrada_saida_manual = models.ForeignKey('EntradaSaidaManual', models.DO_NOTHING, db_column='item_entrada_saida_manual', blank=True, null=True)
    item_emprestei = models.ForeignKey('ProdutosEmprestei', models.DO_NOTHING, db_column='item_emprestei', blank=True, null=True)
    item_emprestei_devolvi = models.ForeignKey('ProdutosEmpresteiDevolvi', models.DO_NOTHING, db_column='item_emprestei_devolvi', blank=True, null=True)
    item_emprestado = models.ForeignKey('ProdutosEmprestados', models.DO_NOTHING, db_column='item_emprestado', blank=True, null=True)
    item_emprestado_devolvido = models.ForeignKey('ProdutosEmprestadosDevolvidos', models.DO_NOTHING, db_column='item_emprestado_devolvido', blank=True, null=True)
    item_remessa_conserto_entrada = models.ForeignKey('ItensRemessaEntrada', models.DO_NOTHING, db_column='item_remessa_conserto_entrada', blank=True, null=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto')
    quantidade = models.DecimalField(max_digits=20, decimal_places=3)
    pai = models.ForeignKey('self', models.DO_NOTHING, db_column='pai', blank=True, null=True)
    possui_filho = models.BooleanField()
    item_ordem_servico_despesas = models.ForeignKey(CabecalhoOrdemServicoDespesas, models.DO_NOTHING, db_column='item_ordem_servico_despesas', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'composicao_movimento'


class ComprasRelacaoYandeh(models.Model):
    codigo = models.AutoField(primary_key=True)
    compra = models.ForeignKey(CabecalhoNotaCompra, models.DO_NOTHING, db_column='compra', blank=True, null=True)
    id_yandeh = models.CharField(max_length=20, blank=True, null=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador', blank=True, null=True)
    data = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'compras_relacao_yandeh'


class Condicional(models.Model):
    codigo = models.AutoField(primary_key=True)
    produto = models.CharField(max_length=100, blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    data_mov = models.DateField(blank=True, null=True)
    vendedor = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='vendedor', blank=True, null=True)
    cd_cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cd_cliente', blank=True, null=True)
    cod_observacao = models.ForeignKey('ObservacoesMovimentos', models.DO_NOTHING, db_column='cod_observacao', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'condicional'


class ConfBoletos(models.Model):
    banco = models.OneToOneField(Bancos, models.DO_NOTHING, db_column='banco', primary_key=True)
    lin_inicio = models.IntegerField(blank=True, null=True)
    entre_boletos = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'conf_boletos'


class ConfBoletosDll(models.Model):
    codigo = models.AutoField(primary_key=True)
    lin_inicio = models.IntegerField(blank=True, null=True)
    entre_boletos = models.IntegerField(blank=True, null=True)
    convenio = models.ForeignKey('ConvenioBancos', models.DO_NOTHING, db_column='convenio', blank=True, null=True)
    aceite = models.IntegerField(blank=True, null=True)
    modeloimpressao = models.IntegerField(blank=True, null=True)
    observacao = models.CharField(max_length=1024, blank=True, null=True)
    usarendereco = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'conf_boletos_dll'


class ConfExterna(models.Model):
    qtd = models.DecimalField(max_digits=65535, decimal_places=65535)
    codigo = models.AutoField(primary_key=True)
    item_ordem_servico = models.ForeignKey('ItensOrdemServico', models.DO_NOTHING, db_column='item_ordem_servico', blank=True, null=True)
    item_nota_compra = models.ForeignKey('ItensNotaCompra', models.DO_NOTHING, db_column='item_nota_compra', blank=True, null=True)
    item_transferencia = models.ForeignKey('ItensTransferencia', models.DO_NOTHING, db_column='item_transferencia', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'conf_externa'


class ConferenciaMesaImpressa(models.Model):
    codigo = models.AutoField(primary_key=True)
    ordem = models.ForeignKey(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem')
    ecf = models.ForeignKey('Terminais', models.DO_NOTHING, db_column='ecf')
    cer = models.IntegerField()
    coo = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'conferencia_mesa_impressa'


class Conffiscal(models.Model):
    codigo = models.AutoField(primary_key=True)
    modelo = models.CharField(max_length=3, blank=True, null=True)
    serie = models.CharField(max_length=3, blank=True, null=True)
    descricao = models.CharField(max_length=30, blank=True, null=True)
    altura = models.IntegerField(blank=True, null=True)
    colunas = models.IntegerField(blank=True, null=True)
    numero = models.BooleanField(blank=True, null=True)
    numerolinha = models.IntegerField(blank=True, null=True)
    numerocoluna = models.IntegerField(blank=True, null=True)
    numerocanhoto = models.BooleanField(blank=True, null=True)
    numerocanhotolinha = models.IntegerField(blank=True, null=True)
    numerocanhotocoluna = models.IntegerField(blank=True, null=True)
    saida = models.BooleanField(blank=True, null=True)
    saidalinha = models.IntegerField(blank=True, null=True)
    saidacoluna = models.IntegerField(blank=True, null=True)
    entrada = models.BooleanField(blank=True, null=True)
    entradalinha = models.IntegerField(blank=True, null=True)
    entradacoluna = models.IntegerField(blank=True, null=True)
    emissorendereco = models.BooleanField(blank=True, null=True)
    emissorenderecolinha = models.IntegerField(blank=True, null=True)
    emissorenderecocoluna = models.IntegerField(blank=True, null=True)
    emissorcidade = models.BooleanField(blank=True, null=True)
    emissorcidadelinha = models.IntegerField(blank=True, null=True)
    emissorcidadecoluna = models.IntegerField(blank=True, null=True)
    emissorcep = models.BooleanField(blank=True, null=True)
    emissorceplinha = models.IntegerField(blank=True, null=True)
    emissorcepcoluna = models.IntegerField(blank=True, null=True)
    emissoruf = models.BooleanField(blank=True, null=True)
    emissoruflinha = models.IntegerField(blank=True, null=True)
    emissorufcoluna = models.IntegerField(blank=True, null=True)
    emissorfone = models.BooleanField(blank=True, null=True)
    emissorfonelinha = models.IntegerField(blank=True, null=True)
    emissorfonecoluna = models.IntegerField(blank=True, null=True)
    emissorcnpj = models.BooleanField(blank=True, null=True)
    emissorcnpjlinha = models.IntegerField(blank=True, null=True)
    emissorcnpjcoluna = models.IntegerField(blank=True, null=True)
    emissorie = models.BooleanField(blank=True, null=True)
    emissorielinha = models.IntegerField(blank=True, null=True)
    emissoriecoluna = models.IntegerField(blank=True, null=True)
    emissorvendedor = models.BooleanField(blank=True, null=True)
    emissorvendedorlinha = models.IntegerField(blank=True, null=True)
    emissorvendedorcoluna = models.IntegerField(blank=True, null=True)
    emissorresponsavel = models.BooleanField(blank=True, null=True)
    emissorresponsavellinha = models.IntegerField(blank=True, null=True)
    emissorresponsavelcoluna = models.IntegerField(blank=True, null=True)
    emissorconferente = models.BooleanField(blank=True, null=True)
    emissorconferentelinha = models.IntegerField(blank=True, null=True)
    emissorconferentecoluna = models.IntegerField(blank=True, null=True)
    natureza = models.BooleanField(blank=True, null=True)
    naturezalinha = models.IntegerField(blank=True, null=True)
    naturezacoluna = models.IntegerField(blank=True, null=True)
    cfop = models.BooleanField(blank=True, null=True)
    cfoplinha = models.IntegerField(blank=True, null=True)
    cfopcoluna = models.IntegerField(blank=True, null=True)
    iesubstrib = models.BooleanField(blank=True, null=True)
    iesubstriblinha = models.IntegerField(blank=True, null=True)
    iesubstribcoluna = models.IntegerField(blank=True, null=True)
    nome = models.BooleanField(blank=True, null=True)
    nomelinha = models.IntegerField(blank=True, null=True)
    nomecoluna = models.IntegerField(blank=True, null=True)
    cnpj = models.BooleanField(blank=True, null=True)
    cnpjlinha = models.IntegerField(blank=True, null=True)
    cnpjcoluna = models.IntegerField(blank=True, null=True)
    endereco = models.BooleanField(blank=True, null=True)
    enderecolinha = models.IntegerField(blank=True, null=True)
    enderecocoluna = models.IntegerField(blank=True, null=True)
    bairro = models.BooleanField(blank=True, null=True)
    bairrolinha = models.IntegerField(blank=True, null=True)
    bairrocoluna = models.IntegerField(blank=True, null=True)
    cep = models.BooleanField(blank=True, null=True)
    ceplinha = models.IntegerField(blank=True, null=True)
    cepcoluna = models.IntegerField(blank=True, null=True)
    cidade = models.BooleanField(blank=True, null=True)
    cidadelinha = models.IntegerField(blank=True, null=True)
    cidadecoluna = models.IntegerField(blank=True, null=True)
    fonefax = models.BooleanField(blank=True, null=True)
    fonefaxlinha = models.IntegerField(blank=True, null=True)
    fonefaxcoluna = models.IntegerField(blank=True, null=True)
    uf = models.BooleanField(blank=True, null=True)
    uflinha = models.IntegerField(blank=True, null=True)
    ufcoluna = models.IntegerField(blank=True, null=True)
    inscrest = models.BooleanField(blank=True, null=True)
    inscrestlinha = models.IntegerField(blank=True, null=True)
    inscrestcoluna = models.IntegerField(blank=True, null=True)
    dataemissao = models.BooleanField(blank=True, null=True)
    dataemissaolinha = models.IntegerField(blank=True, null=True)
    dataemissaocoluna = models.IntegerField(blank=True, null=True)
    datasaida = models.BooleanField(blank=True, null=True)
    datasaidalinha = models.IntegerField(blank=True, null=True)
    datasaidacoluna = models.IntegerField(blank=True, null=True)
    horasaida = models.BooleanField(blank=True, null=True)
    horasaidalinha = models.IntegerField(blank=True, null=True)
    horasaidacoluna = models.IntegerField(blank=True, null=True)
    codigoproduto = models.BooleanField(blank=True, null=True)
    codigoprodutolinha = models.IntegerField(blank=True, null=True)
    codigoprodutocoluna = models.IntegerField(blank=True, null=True)
    descricaoproduto = models.BooleanField(blank=True, null=True)
    descricaoprodutolinha = models.IntegerField(blank=True, null=True)
    descricaoprodutocoluna = models.IntegerField(blank=True, null=True)
    sittrib = models.BooleanField(blank=True, null=True)
    sittriblinha = models.IntegerField(blank=True, null=True)
    sittribcoluna = models.IntegerField(blank=True, null=True)
    cf = models.BooleanField(blank=True, null=True)
    cflinha = models.IntegerField(blank=True, null=True)
    cfcoluna = models.IntegerField(blank=True, null=True)
    unidade = models.BooleanField(blank=True, null=True)
    unidadelinha = models.IntegerField(blank=True, null=True)
    unidadecoluna = models.IntegerField(blank=True, null=True)
    quantidade = models.BooleanField(blank=True, null=True)
    quantidadelinha = models.IntegerField(blank=True, null=True)
    quantidadecoluna = models.IntegerField(blank=True, null=True)
    valorunita = models.BooleanField(blank=True, null=True)
    valorunitalinha = models.IntegerField(blank=True, null=True)
    valorunitacoluna = models.IntegerField(blank=True, null=True)
    valortotal = models.BooleanField(blank=True, null=True)
    valortotallinha = models.IntegerField(blank=True, null=True)
    valortotalcoluna = models.IntegerField(blank=True, null=True)
    icms = models.BooleanField(blank=True, null=True)
    icmslinha = models.IntegerField(blank=True, null=True)
    icmscoluna = models.IntegerField(blank=True, null=True)
    iss = models.BooleanField(blank=True, null=True)
    isslinha = models.IntegerField(blank=True, null=True)
    isscoluna = models.IntegerField(blank=True, null=True)
    ipi = models.BooleanField(blank=True, null=True)
    ipilinha = models.IntegerField(blank=True, null=True)
    ipicoluna = models.IntegerField(blank=True, null=True)
    totalbruto = models.BooleanField(blank=True, null=True)
    totalbrutolinha = models.IntegerField(blank=True, null=True)
    totalbrutocoluna = models.IntegerField(blank=True, null=True)
    desconto = models.BooleanField(blank=True, null=True)
    descontolinha = models.IntegerField(blank=True, null=True)
    descontocoluna = models.IntegerField(blank=True, null=True)
    baseicms = models.BooleanField(blank=True, null=True)
    baseicmslinha = models.IntegerField(blank=True, null=True)
    baseicmscoluna = models.IntegerField(blank=True, null=True)
    valoricms = models.BooleanField(blank=True, null=True)
    valoricmslinha = models.IntegerField(blank=True, null=True)
    valoricmscoluna = models.IntegerField(blank=True, null=True)
    baseicmssubst = models.BooleanField(blank=True, null=True)
    baseicmssubstlinha = models.IntegerField(blank=True, null=True)
    baseicmssubstcoluna = models.IntegerField(blank=True, null=True)
    valoricmssubst = models.BooleanField(blank=True, null=True)
    valoricmssubstlinha = models.IntegerField(blank=True, null=True)
    valoricmssubstcoluna = models.IntegerField(blank=True, null=True)
    totalprodutos = models.BooleanField(blank=True, null=True)
    totalprodutoslinha = models.IntegerField(blank=True, null=True)
    totalprodutoscoluna = models.IntegerField(blank=True, null=True)
    valorfrete = models.BooleanField(blank=True, null=True)
    valorfretelinha = models.IntegerField(blank=True, null=True)
    valorfretecoluna = models.IntegerField(blank=True, null=True)
    valorseguro = models.BooleanField(blank=True, null=True)
    valorsegurolinha = models.IntegerField(blank=True, null=True)
    valorsegurocoluna = models.IntegerField(blank=True, null=True)
    valoroutras = models.BooleanField(blank=True, null=True)
    valoroutraslinha = models.IntegerField(blank=True, null=True)
    valoroutrascoluna = models.IntegerField(blank=True, null=True)
    valoripi = models.BooleanField(blank=True, null=True)
    valoripilinha = models.IntegerField(blank=True, null=True)
    valoripicoluna = models.IntegerField(blank=True, null=True)
    totalnota = models.BooleanField(blank=True, null=True)
    totalnotalinha = models.IntegerField(blank=True, null=True)
    totalnotacoluna = models.IntegerField(blank=True, null=True)
    transportnome = models.BooleanField(blank=True, null=True)
    transportnomelinha = models.IntegerField(blank=True, null=True)
    transportnomecoluna = models.IntegerField(blank=True, null=True)
    freteconta = models.BooleanField(blank=True, null=True)
    fretecontalinha = models.IntegerField(blank=True, null=True)
    fretecontacoluna = models.IntegerField(blank=True, null=True)
    placa = models.BooleanField(blank=True, null=True)
    placalinha = models.IntegerField(blank=True, null=True)
    placacoluna = models.IntegerField(blank=True, null=True)
    transportufveiculo = models.BooleanField(blank=True, null=True)
    transportufveiculolinha = models.IntegerField(blank=True, null=True)
    transportufveiculocoluna = models.IntegerField(blank=True, null=True)
    transportcnpjcpf = models.BooleanField(blank=True, null=True)
    transportcnpjcpflinha = models.IntegerField(blank=True, null=True)
    transportcnpjcpfcoluna = models.IntegerField(blank=True, null=True)
    transportendereco = models.BooleanField(blank=True, null=True)
    transportenderecolinha = models.IntegerField(blank=True, null=True)
    transportenderecocoluna = models.IntegerField(blank=True, null=True)
    transportcidade = models.BooleanField(blank=True, null=True)
    transportcidadelinha = models.IntegerField(blank=True, null=True)
    transportcidadecoluna = models.IntegerField(blank=True, null=True)
    transportuf = models.BooleanField(blank=True, null=True)
    transportuflinha = models.IntegerField(blank=True, null=True)
    transportufcoluna = models.IntegerField(blank=True, null=True)
    transportie = models.BooleanField(blank=True, null=True)
    transportielinha = models.IntegerField(blank=True, null=True)
    transportiecoluna = models.IntegerField(blank=True, null=True)
    transportquantidade = models.BooleanField(blank=True, null=True)
    transportquantidadelinha = models.IntegerField(blank=True, null=True)
    transportquantidadecoluna = models.IntegerField(blank=True, null=True)
    transportespecie = models.BooleanField(blank=True, null=True)
    transportespecielinha = models.IntegerField(blank=True, null=True)
    transportespeciecoluna = models.IntegerField(blank=True, null=True)
    transportmarca = models.BooleanField(blank=True, null=True)
    transportmarcalinha = models.IntegerField(blank=True, null=True)
    transportmarcacoluna = models.IntegerField(blank=True, null=True)
    transportnumero = models.BooleanField(blank=True, null=True)
    transportnumerolinha = models.IntegerField(blank=True, null=True)
    transportnumerocoluna = models.IntegerField(blank=True, null=True)
    pesobruto = models.BooleanField(blank=True, null=True)
    pesobrutolinha = models.IntegerField(blank=True, null=True)
    pesobrutocoluna = models.IntegerField(blank=True, null=True)
    pesoliquido = models.BooleanField(blank=True, null=True)
    pesoliquidolinha = models.IntegerField(blank=True, null=True)
    pesoliquidocoluna = models.IntegerField(blank=True, null=True)
    dados = models.BooleanField(blank=True, null=True)
    dadoslinha = models.IntegerField(blank=True, null=True)
    dadoscoluna = models.IntegerField(blank=True, null=True)
    itens = models.IntegerField(blank=True, null=True)
    larguradados = models.IntegerField(blank=True, null=True)
    redespnome = models.BooleanField(blank=True, null=True)
    redespnomelinha = models.IntegerField(blank=True, null=True)
    redespnomecoluna = models.IntegerField(blank=True, null=True)
    redespbairro = models.BooleanField(blank=True, null=True)
    redespbairrolinha = models.IntegerField(blank=True, null=True)
    redespbairrocoluna = models.IntegerField(blank=True, null=True)
    redespfone = models.BooleanField(blank=True, null=True)
    redespfonelinha = models.IntegerField(blank=True, null=True)
    redespfonecoluna = models.IntegerField(blank=True, null=True)
    redespfreteconta = models.BooleanField(blank=True, null=True)
    redespfretecontalinha = models.IntegerField(blank=True, null=True)
    redespfretecontacoluna = models.IntegerField(blank=True, null=True)
    redespplaca = models.BooleanField(blank=True, null=True)
    redespplacalinha = models.IntegerField(blank=True, null=True)
    redespplacacoluna = models.IntegerField(blank=True, null=True)
    redespufveiculo = models.BooleanField(blank=True, null=True)
    redespufveiculolinha = models.IntegerField(blank=True, null=True)
    redespufveiculocoluna = models.IntegerField(blank=True, null=True)
    redespcnpjcpf = models.BooleanField(blank=True, null=True)
    redespcnpjcpflinha = models.IntegerField(blank=True, null=True)
    redespcnpjcpfcoluna = models.IntegerField(blank=True, null=True)
    redespendereco = models.BooleanField(blank=True, null=True)
    redespenderecolinha = models.IntegerField(blank=True, null=True)
    redespenderecocoluna = models.IntegerField(blank=True, null=True)
    redespcidade = models.BooleanField(blank=True, null=True)
    redespcidadelinha = models.IntegerField(blank=True, null=True)
    redespcidadecoluna = models.IntegerField(blank=True, null=True)
    redespuf = models.BooleanField(blank=True, null=True)
    redespuflinha = models.IntegerField(blank=True, null=True)
    redespufcoluna = models.IntegerField(blank=True, null=True)
    redespie = models.BooleanField(blank=True, null=True)
    redespielinha = models.IntegerField(blank=True, null=True)
    redespiecoluna = models.IntegerField(blank=True, null=True)
    redespquantidade = models.BooleanField(blank=True, null=True)
    redespquantidadelinha = models.IntegerField(blank=True, null=True)
    redespquantidadecoluna = models.IntegerField(blank=True, null=True)
    redespespecie = models.BooleanField(blank=True, null=True)
    redespespecielinha = models.IntegerField(blank=True, null=True)
    redespespeciecoluna = models.IntegerField(blank=True, null=True)
    redespmarca = models.BooleanField(blank=True, null=True)
    redespmarcalinha = models.IntegerField(blank=True, null=True)
    redespmarcacoluna = models.IntegerField(blank=True, null=True)
    redespnumero = models.BooleanField(blank=True, null=True)
    redespnumerolinha = models.IntegerField(blank=True, null=True)
    redespnumerocoluna = models.IntegerField(blank=True, null=True)
    servdescricao = models.BooleanField(blank=True, null=True)
    servdescricaolinha = models.IntegerField(blank=True, null=True)
    servdescricaocoluna = models.IntegerField(blank=True, null=True)
    servbaseiss = models.BooleanField(blank=True, null=True)
    servbaseisslinha = models.IntegerField(blank=True, null=True)
    servbaseisscoluna = models.IntegerField(blank=True, null=True)
    servaliquota = models.BooleanField(blank=True, null=True)
    servaliquotalinha = models.IntegerField(blank=True, null=True)
    servaliquotacoluna = models.IntegerField(blank=True, null=True)
    servvaloriss = models.BooleanField(blank=True, null=True)
    servvalorisslinha = models.IntegerField(blank=True, null=True)
    servvalorisscoluna = models.IntegerField(blank=True, null=True)
    servinscmunicipal = models.BooleanField(blank=True, null=True)
    servinscmunicipallinha = models.IntegerField(blank=True, null=True)
    servinscmunicipalcoluna = models.IntegerField(blank=True, null=True)
    servtotal = models.BooleanField(blank=True, null=True)
    servtotallinha = models.IntegerField(blank=True, null=True)
    servtotalcoluna = models.IntegerField(blank=True, null=True)
    descontonota = models.BooleanField(blank=True, null=True)
    descontonotalinha = models.IntegerField(blank=True, null=True)
    descontonotacoluna = models.IntegerField(blank=True, null=True)
    servdesconto = models.BooleanField(blank=True, null=True)
    servdescontolinha = models.IntegerField(blank=True, null=True)
    servdescontocoluna = models.IntegerField(blank=True, null=True)
    servcodigo = models.BooleanField(blank=True, null=True)
    servcodigolinha = models.IntegerField(blank=True, null=True)
    servcodigocoluna = models.IntegerField(blank=True, null=True)
    servunidade = models.BooleanField(blank=True, null=True)
    servunidadelinha = models.IntegerField(blank=True, null=True)
    servunidadecoluna = models.IntegerField(blank=True, null=True)
    servqtd = models.BooleanField(blank=True, null=True)
    servqtdlinha = models.IntegerField(blank=True, null=True)
    servqtdcoluna = models.IntegerField(blank=True, null=True)
    servunitario = models.BooleanField(blank=True, null=True)
    servunitariolinha = models.IntegerField(blank=True, null=True)
    servunitariocoluna = models.IntegerField(blank=True, null=True)
    servtotalitem = models.BooleanField(blank=True, null=True)
    servtotalitemlinha = models.IntegerField(blank=True, null=True)
    servtotalitemcoluna = models.IntegerField(blank=True, null=True)
    servnumeroitens = models.IntegerField(blank=True, null=True)
    espacamento = models.IntegerField(blank=True, null=True)
    dupla_passada = models.BooleanField(blank=True, null=True)
    redesppesobruto = models.BooleanField(blank=True, null=True)
    redesppesobrutolinha = models.IntegerField(blank=True, null=True)
    redesppesobrutocoluna = models.IntegerField(blank=True, null=True)
    redesppesoliq = models.BooleanField(blank=True, null=True)
    redesppesoliqlinha = models.IntegerField(blank=True, null=True)
    redesppesoliqcoluna = models.IntegerField(blank=True, null=True)
    xentrada = models.BooleanField(blank=True, null=True)
    xentradalinha = models.IntegerField(blank=True, null=True)
    xentradacoluna = models.IntegerField(blank=True, null=True)
    xsaida = models.BooleanField(blank=True, null=True)
    xsaidalinha = models.IntegerField(blank=True, null=True)
    xsaidacoluna = models.IntegerField(blank=True, null=True)
    fatura1numero = models.BooleanField(blank=True, null=True)
    fatura1numerolinha = models.IntegerField(blank=True, null=True)
    fatura1numerocoluna = models.IntegerField(blank=True, null=True)
    fatura1valor = models.BooleanField(blank=True, null=True)
    fatura1valorlinha = models.IntegerField(blank=True, null=True)
    fatura1valorcoluna = models.IntegerField(blank=True, null=True)
    fatura1data = models.BooleanField(blank=True, null=True)
    fatura1datalinha = models.IntegerField(blank=True, null=True)
    fatura1datacoluna = models.IntegerField(blank=True, null=True)
    fatura2numero = models.BooleanField(blank=True, null=True)
    fatura2numerolinha = models.IntegerField(blank=True, null=True)
    fatura2numerocoluna = models.IntegerField(blank=True, null=True)
    fatura2valor = models.BooleanField(blank=True, null=True)
    fatura2valorlinha = models.IntegerField(blank=True, null=True)
    fatura2valorcoluna = models.IntegerField(blank=True, null=True)
    fatura2data = models.BooleanField(blank=True, null=True)
    fatura2datalinha = models.IntegerField(blank=True, null=True)
    fatura2datacoluna = models.IntegerField(blank=True, null=True)
    fatura3numero = models.BooleanField(blank=True, null=True)
    fatura3numerolinha = models.IntegerField(blank=True, null=True)
    fatura3numerocoluna = models.IntegerField(blank=True, null=True)
    fatura3valor = models.BooleanField(blank=True, null=True)
    fatura3valorlinha = models.IntegerField(blank=True, null=True)
    fatura3valorcoluna = models.IntegerField(blank=True, null=True)
    fatura3data = models.BooleanField(blank=True, null=True)
    fatura3datalinha = models.IntegerField(blank=True, null=True)
    fatura3datacoluna = models.IntegerField(blank=True, null=True)
    fatura4numero = models.BooleanField(blank=True, null=True)
    fatura4numerolinha = models.IntegerField(blank=True, null=True)
    fatura4numerocoluna = models.IntegerField(blank=True, null=True)
    fatura4valor = models.BooleanField(blank=True, null=True)
    fatura4valorlinha = models.IntegerField(blank=True, null=True)
    fatura4valorcoluna = models.IntegerField(blank=True, null=True)
    fatura4data = models.BooleanField(blank=True, null=True)
    fatura4datalinha = models.IntegerField(blank=True, null=True)
    fatura4datacoluna = models.IntegerField(blank=True, null=True)
    fatura5numero = models.BooleanField(blank=True, null=True)
    fatura5numerolinha = models.IntegerField(blank=True, null=True)
    fatura5numerocoluna = models.IntegerField(blank=True, null=True)
    fatura5valor = models.BooleanField(blank=True, null=True)
    fatura5valorlinha = models.IntegerField(blank=True, null=True)
    fatura5valorcoluna = models.IntegerField(blank=True, null=True)
    fatura5data = models.BooleanField(blank=True, null=True)
    fatura5datalinha = models.IntegerField(blank=True, null=True)
    fatura5datacoluna = models.IntegerField(blank=True, null=True)
    fatura6numero = models.BooleanField(blank=True, null=True)
    fatura6numerolinha = models.IntegerField(blank=True, null=True)
    fatura6numerocoluna = models.IntegerField(blank=True, null=True)
    fatura6valor = models.BooleanField(blank=True, null=True)
    fatura6valorlinha = models.IntegerField(blank=True, null=True)
    fatura6valorcoluna = models.IntegerField(blank=True, null=True)
    fatura6data = models.BooleanField(blank=True, null=True)
    fatura6datalinha = models.IntegerField(blank=True, null=True)
    fatura6datacoluna = models.IntegerField(blank=True, null=True)
    larguraitens = models.IntegerField(blank=True, null=True)
    cfop_item = models.BooleanField(blank=True, null=True)
    cfop_itemlinha = models.IntegerField(blank=True, null=True)
    cfop_itemcoluna = models.IntegerField(blank=True, null=True)
    cfop_serv = models.BooleanField(blank=True, null=True)
    cfop_servlinha = models.IntegerField(blank=True, null=True)
    cfop_servcoluna = models.IntegerField(blank=True, null=True)
    porcemdescontoi = models.BooleanField(blank=True, null=True)
    porcemdescontoilinha = models.IntegerField(blank=True, null=True)
    porcemdescontoicoluna = models.IntegerField(blank=True, null=True)
    porcemdescontos = models.BooleanField(blank=True, null=True)
    porcemdescontoslinha = models.IntegerField(blank=True, null=True)
    porcemdescontoscoluna = models.IntegerField(blank=True, null=True)
    ipiitem = models.BooleanField(blank=True, null=True)
    ipiitemlinha = models.IntegerField(blank=True, null=True)
    ipiitemcoluna = models.IntegerField(blank=True, null=True)
    larguraserv = models.IntegerField(blank=True, null=True)
    larguranota = models.IntegerField(blank=True, null=True)
    baseiss = models.BooleanField(blank=True, null=True)
    baseisslinha = models.IntegerField(blank=True, null=True)
    baseisscoluna = models.IntegerField(blank=True, null=True)
    ordem = models.BooleanField(blank=True, null=True)
    ordemlinha = models.IntegerField(blank=True, null=True)
    ordemcoluna = models.IntegerField(blank=True, null=True)
    praca = models.BooleanField(blank=True, null=True)
    pracalinha = models.IntegerField(blank=True, null=True)
    pracacoluna = models.IntegerField(blank=True, null=True)
    condicao = models.BooleanField(blank=True, null=True)
    condicaolinha = models.IntegerField(blank=True, null=True)
    condicaocoluna = models.IntegerField(blank=True, null=True)
    issnota = models.BooleanField(blank=True, null=True)
    issnotalinha = models.IntegerField(blank=True, null=True)
    issnotacoluna = models.IntegerField(blank=True, null=True)
    mascara_valor = models.IntegerField(blank=True, null=True)
    parcela = models.BooleanField(blank=True, null=True)
    texto1 = models.BooleanField(blank=True, null=True)
    texto1linha = models.IntegerField(blank=True, null=True)
    texto1coluna = models.IntegerField(blank=True, null=True)
    texto1_valor = models.CharField(max_length=255, blank=True, null=True)
    texto2 = models.BooleanField(blank=True, null=True)
    texto2linha = models.IntegerField(blank=True, null=True)
    texto2coluna = models.IntegerField(blank=True, null=True)
    texto2_valor = models.CharField(max_length=255, blank=True, null=True)
    texto3 = models.BooleanField(blank=True, null=True)
    texto3linha = models.IntegerField(blank=True, null=True)
    texto3coluna = models.IntegerField(blank=True, null=True)
    texto3_valor = models.CharField(max_length=255, blank=True, null=True)
    texto4 = models.BooleanField(blank=True, null=True)
    texto4linha = models.IntegerField(blank=True, null=True)
    texto4coluna = models.IntegerField(blank=True, null=True)
    texto4_valor = models.CharField(max_length=255, blank=True, null=True)
    nome_vendedor = models.BooleanField(blank=True, null=True)
    nome_vendedorlinha = models.IntegerField(blank=True, null=True)
    nome_vendedorcoluna = models.IntegerField(blank=True, null=True)
    cliente_canhoto = models.BooleanField(blank=True, null=True)
    cliente_canhotolinha = models.IntegerField(blank=True, null=True)
    cliente_canhotocoluna = models.IntegerField(blank=True, null=True)
    total_canhoto = models.BooleanField(blank=True, null=True)
    total_canhotolinha = models.IntegerField(blank=True, null=True)
    total_canhotocoluna = models.IntegerField(blank=True, null=True)
    fantasia = models.BooleanField(blank=True, null=True)
    fantasialinha = models.IntegerField(blank=True, null=True)
    fantasiacoluna = models.IntegerField(blank=True, null=True)
    referencia = models.BooleanField(blank=True, null=True)
    referencialinha = models.IntegerField(blank=True, null=True)
    referenciacoluna = models.IntegerField(blank=True, null=True)
    cli_codigo = models.BooleanField(blank=True, null=True)
    cli_codigolinha = models.IntegerField(blank=True, null=True)
    cli_codigocoluna = models.IntegerField(blank=True, null=True)
    cfop_totais = models.BooleanField(blank=True, null=True)
    cfop_totaislinha = models.IntegerField(blank=True, null=True)
    cfop_totaiscoluna = models.IntegerField(blank=True, null=True)
    tamanho_quantidade = models.IntegerField(blank=True, null=True)
    decimais_quantidade = models.IntegerField(blank=True, null=True)
    tamanho_unitario = models.IntegerField(blank=True, null=True)
    decimais_unitario = models.IntegerField(blank=True, null=True)
    tamanho_total = models.IntegerField(blank=True, null=True)
    decimais_total = models.IntegerField(blank=True, null=True)
    st_totais = models.BooleanField(blank=True, null=True)
    st_totaislinha = models.IntegerField(blank=True, null=True)
    st_totaiscoluna = models.IntegerField(blank=True, null=True)
    valor_extenso_1 = models.BooleanField(blank=True, null=True)
    valor_extenso_1linha = models.IntegerField(blank=True, null=True)
    valor_extenso_1coluna = models.IntegerField(blank=True, null=True)
    valor_extenso_2 = models.BooleanField(blank=True, null=True)
    valor_extenso_2linha = models.IntegerField(blank=True, null=True)
    valor_extenso_2coluna = models.IntegerField(blank=True, null=True)
    funrural_base = models.BooleanField(blank=True, null=True)
    funrural_baselinha = models.IntegerField(blank=True, null=True)
    funrural_basecoluna = models.IntegerField(blank=True, null=True)
    funrural_aliquota = models.BooleanField(blank=True, null=True)
    funrural_aliquotalinha = models.IntegerField(blank=True, null=True)
    funrural_aliquotacoluna = models.IntegerField(blank=True, null=True)
    funrural_valor = models.BooleanField(blank=True, null=True)
    funrural_valorlinha = models.IntegerField(blank=True, null=True)
    funrural_valorcoluna = models.IntegerField(blank=True, null=True)
    codigo_fabrica = models.BooleanField(blank=True, null=True)
    codigo_fabricalinha = models.IntegerField(blank=True, null=True)
    codigo_fabricacoluna = models.IntegerField(blank=True, null=True)
    tamanho_extenso = models.IntegerField(blank=True, null=True)
    canhoto_endereco = models.BooleanField(blank=True, null=True)
    canhoto_enderecolinha = models.IntegerField(blank=True, null=True)
    canhoto_enderecocoluna = models.IntegerField(blank=True, null=True)
    canhoto_cod_cli = models.BooleanField(blank=True, null=True)
    canhoto_cod_clilinha = models.IntegerField(blank=True, null=True)
    canhoto_cod_clicoluna = models.IntegerField(blank=True, null=True)
    canhoto_data = models.BooleanField(blank=True, null=True)
    canhoto_datalinha = models.IntegerField(blank=True, null=True)
    canhoto_datacoluna = models.IntegerField(blank=True, null=True)
    canhoto_peso = models.BooleanField(blank=True, null=True)
    canhoto_pesolinha = models.IntegerField(blank=True, null=True)
    canhoto_pesocoluna = models.IntegerField(blank=True, null=True)
    tamanho_aliquota = models.IntegerField(blank=True, null=True)
    decimais_aliquota = models.IntegerField(blank=True, null=True)
    obs_extra = models.BooleanField(blank=True, null=True)
    embalagem_item = models.BooleanField(blank=True, null=True)
    embalagem_itemlinha = models.IntegerField(blank=True, null=True)
    embalagem_itemcoluna = models.IntegerField(blank=True, null=True)
    coluna_duplicar = models.IntegerField(blank=True, null=True)
    entre_colunas = models.IntegerField(blank=True, null=True)
    numerodigitosncm = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'conffiscal'


class ConfiguracaoItens(models.Model):
    codigo = models.AutoField(primary_key=True)
    cod_configuracao = models.ForeignKey('ConfiguracaoModelo', models.DO_NOTHING, db_column='cod_configuracao', blank=True, null=True)
    referencia = models.CharField(max_length=30, blank=True, null=True)
    linha = models.IntegerField()
    coluna = models.IntegerField()
    alinhamento = models.CharField(max_length=1, blank=True, null=True)
    tipo_dado = models.CharField(max_length=100)
    tamanho = models.IntegerField()
    precisao = models.IntegerField()
    mascara = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'configuracao_itens'


class ConfiguracaoModelo(models.Model):
    codigo = models.AutoField(primary_key=True)
    tipo = models.CharField(max_length=20, blank=True, null=True)
    descricao = models.CharField(max_length=50, blank=True, null=True)
    serie = models.IntegerField(blank=True, null=True)
    altura = models.IntegerField()
    largura = models.IntegerField()
    nro_linhas = models.IntegerField()
    nro_colunas = models.IntegerField()
    espacamento = models.IntegerField()
    compactado = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'configuracao_modelo'


class Configuracoes(models.Model):
    grupo_servico = models.IntegerField(blank=True, null=True)
    grupo_servico2 = models.IntegerField(blank=True, null=True)
    grupo_consumo = models.IntegerField(blank=True, null=True)
    grupo_orcamento = models.IntegerField(blank=True, null=True)
    dataparcelas = models.BooleanField(blank=True, null=True)
    data_apagar_emissao = models.BooleanField(blank=True, null=True)
    faturando = models.BooleanField()
    passando_manutencao = models.BooleanField()
    atualizando = models.BooleanField()
    lanca_quantidade_total_valor_producao = models.BooleanField()
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'configuracoes'


class Configuracoes2(models.Model):
    campo1 = models.CharField(max_length=-1, blank=True, null=True)
    campo2 = models.CharField(max_length=-1, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'configuracoes2'


class Conjuge(models.Model):
    codigo = models.AutoField(primary_key=True)
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente', blank=True, null=True)
    nome = models.CharField(max_length=50, blank=True, null=True)
    cpf = models.CharField(max_length=14, blank=True, null=True)
    rg = models.CharField(max_length=20, blank=True, null=True)
    datanasc = models.DateField(blank=True, null=True)
    empresa_trabalha = models.CharField(max_length=50, blank=True, null=True)
    fone = models.CharField(max_length=14, blank=True, null=True)
    renda = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    data_admissao = models.DateField(blank=True, null=True)
    endereco = models.CharField(max_length=60, blank=True, null=True)
    numero = models.CharField(max_length=10, blank=True, null=True)
    complemento = models.CharField(max_length=100, blank=True, null=True)
    bairro = models.CharField(max_length=60, blank=True, null=True)
    cep = models.CharField(max_length=9, blank=True, null=True)
    ramal = models.IntegerField(blank=True, null=True)
    classe_profissional = models.ForeignKey(ClassesProfissionais, models.DO_NOTHING, db_column='classe_profissional', blank=True, null=True)
    cd_profissao = models.ForeignKey('Profissoes', models.DO_NOTHING, db_column='cd_profissao', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'conjuge'


class Conservacoes(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nome = models.CharField(max_length=100, blank=True, null=True)
    conservacao1 = models.CharField(max_length=56, blank=True, null=True)
    conservacao2 = models.CharField(max_length=56, blank=True, null=True)
    conservacao3 = models.CharField(max_length=56, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'conservacoes'


class ConsultasSpc(models.Model):
    codigo = models.IntegerField(primary_key=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente', blank=True, null=True)
    tipo_consulta = models.CharField(max_length=1)
    data_operacao = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'consultas_spc'


class ContasFolha(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nome = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'contas_folha'


class ContratosImpressos(models.Model):
    ordem = models.ForeignKey(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'contratos_impressos'


class ContratosInternet(models.Model):
    codigo = models.AutoField(primary_key=True)
    velocidade = models.ForeignKey('VelocidadeInternet', models.DO_NOTHING, db_column='velocidade', blank=True, null=True)
    patrimonio = models.CharField(max_length=20, blank=True, null=True)
    antena_particular = models.BooleanField(blank=True, null=True)
    antena_local = models.BooleanField(blank=True, null=True)
    ppoe = models.CharField(max_length=30, blank=True, null=True)
    wan = models.CharField(max_length=15, blank=True, null=True)
    contrato = models.ForeignKey('VigenciaContrato', models.DO_NOTHING, db_column='contrato', blank=True, null=True)
    observacoes = models.TextField(blank=True, null=True)
    pontos_acesso = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'contratos_internet'


class ControleVisitas(models.Model):
    codigo = models.AutoField(primary_key=True)
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente', blank=True, null=True)
    vendedor = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='vendedor', blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    km_inicial = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    km_final = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    contato = models.CharField(max_length=100, blank=True, null=True)
    obs = models.TextField(blank=True, null=True)
    gerou_venda = models.BooleanField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'controle_visitas'


class ConvenioBancos(models.Model):
    codigo = models.AutoField(primary_key=True)
    conta = models.ForeignKey('Ctabancarias', models.DO_NOTHING, db_column='conta', blank=True, null=True)
    localpgto = models.CharField(max_length=150, blank=True, null=True)
    cedente = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    dvcedente = models.CharField(max_length=5, blank=True, null=True)
    carteira = models.IntegerField(blank=True, null=True)
    convenio = models.IntegerField(blank=True, null=True)
    variacao = models.IntegerField(blank=True, null=True)
    gera_sequencial = models.BooleanField()
    sigcb = models.BooleanField()
    carteira_sigla = models.CharField(max_length=10)
    convenio_unicred = models.BooleanField()
    ativo = models.BooleanField()
    agencia_cnpj = models.CharField(max_length=18, blank=True, null=True)
    boletofinal = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    boletoinicial = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    endereco_agencia = models.CharField(max_length=50, blank=True, null=True)
    numero_agencia = models.CharField(max_length=20, blank=True, null=True)
    bairro_agencia = models.CharField(max_length=50, blank=True, null=True)
    cep_agencia = models.ForeignKey(Cidades, models.DO_NOTHING, db_column='cep_agencia', blank=True, null=True)
    adicional = models.CharField(max_length=100, blank=True, null=True)
    adicional2 = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'convenio_bancos'


class CreditoClientes(models.Model):
    codigo = models.AutoField(primary_key=True)
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente', blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    historico = models.CharField(max_length=100, blank=True, null=True)
    entrada = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    saida = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    origem = models.CharField(max_length=20, blank=True, null=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'credito_clientes'


class CreditoFornecedores(models.Model):
    codigo = models.AutoField(primary_key=True)
    fornecedor = models.ForeignKey('Fornecedores', models.DO_NOTHING, db_column='fornecedor', blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    historico = models.CharField(max_length=100, blank=True, null=True)
    entrada = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    saida = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    origem = models.CharField(max_length=20, blank=True, null=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador', blank=True, null=True)
    devolucao_fornecedor = models.ForeignKey('Remessa', models.DO_NOTHING, db_column='devolucao_fornecedor', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'credito_fornecedores'


class CstEnquadramentoValido(models.Model):
    cst = models.ForeignKey(CodigosFiscaisIpi, models.DO_NOTHING, db_column='cst')
    enquadramento = models.ForeignKey(CodigosFiscaisIpiEnquadramento, models.DO_NOTHING, db_column='enquadramento')

    class Meta:
        managed = False
        db_table = 'cst_enquadramento_valido'


class Ctabancarias(models.Model):
    conta = models.CharField(primary_key=True, max_length=13)
    banco = models.ForeignKey(Bancos, models.DO_NOTHING, db_column='banco', blank=True, null=True)
    nome = models.CharField(max_length=50, blank=True, null=True)
    cedente = models.DecimalField(max_digits=11, decimal_places=0, blank=True, null=True)
    limite = models.DecimalField(max_digits=10, decimal_places=0, blank=True, null=True)
    e_caixa_cofre = models.BooleanField(blank=True, null=True)
    localpgto = models.CharField(max_length=50, blank=True, null=True)
    dvcedente = models.IntegerField(blank=True, null=True)
    carteira = models.IntegerField(blank=True, null=True)
    ativo = models.BooleanField()
    cta_nfce_contingencia = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'ctabancarias'


class DadosContabilista(models.Model):
    nome = models.CharField(max_length=60)
    cpf = models.CharField(primary_key=True, max_length=14)
    crc = models.CharField(max_length=15)
    cnpj = models.CharField(max_length=18)
    cep = models.CharField(max_length=9)
    endereco = models.CharField(max_length=60)
    numero = models.CharField(max_length=20)
    complemento = models.CharField(max_length=60, blank=True, null=True)
    bairro = models.CharField(max_length=60)
    fone = models.CharField(max_length=14)
    fax = models.CharField(max_length=14, blank=True, null=True)
    email = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'dados_contabilista'


class DadosFinanceirosCliente(models.Model):
    cliente = models.OneToOneField(Clientes, models.DO_NOTHING, db_column='cliente', primary_key=True)
    pagaalugue = models.CharField(max_length=1, blank=True, null=True)
    valoraluguel = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    renda = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    credito = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    desconto = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    diapgto = models.IntegerField()
    dependentes = models.IntegerField()
    acrescimo_parcela = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    dias_atrazo = models.IntegerField(blank=True, null=True)
    senha = models.CharField(max_length=20)

    class Meta:
        managed = False
        db_table = 'dados_financeiros_cliente'


class DadosSpc(models.Model):
    cliente = models.OneToOneField(Clientes, models.DO_NOTHING, db_column='cliente', primary_key=True)
    nomepai = models.CharField(max_length=50, blank=True, null=True)
    nomemae = models.CharField(max_length=50, blank=True, null=True)
    datanasc = models.DateField(blank=True, null=True)
    negativado = models.CharField(max_length=1, blank=True, null=True)
    datanegativa = models.DateField(blank=True, null=True)
    tanospc = models.CharField(max_length=1, blank=True, null=True)
    naturalidade = models.CharField(max_length=30, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'dados_spc'


class DadosTrabalhoCliente(models.Model):
    cliente = models.OneToOneField(Clientes, models.DO_NOTHING, db_column='cliente', primary_key=True)
    empresa = models.CharField(max_length=50, blank=True, null=True)
    endereco = models.CharField(max_length=50, blank=True, null=True)
    bairro = models.CharField(max_length=30, blank=True, null=True)
    cep = models.ForeignKey(Cidades, models.DO_NOTHING, db_column='cep', blank=True, null=True)
    fone = models.CharField(max_length=15, blank=True, null=True)
    dataadmissao = models.DateField(blank=True, null=True)
    ponto_referencia = models.CharField(max_length=50, blank=True, null=True)
    encarrega = models.CharField(max_length=50, blank=True, null=True)
    setor = models.CharField(max_length=50, blank=True, null=True)
    cracha = models.CharField(max_length=20, blank=True, null=True)
    classe_profissional = models.ForeignKey(ClassesProfissionais, models.DO_NOTHING, db_column='classe_profissional', blank=True, null=True)
    cd_profissao = models.ForeignKey('Profissoes', models.DO_NOTHING, db_column='cd_profissao', blank=True, null=True)
    horario = models.CharField(max_length=20, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'dados_trabalho_cliente'


class Departamentos(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nome = models.CharField(max_length=50, blank=True, null=True)
    aplicacao = models.CharField(max_length=-1, blank=True, null=True)
    observacao = models.CharField(max_length=-1, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'departamentos'


class Desonerado(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    nome = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'desonerado'


class DesoneradoConfiguracao(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    desonerado = models.ForeignKey(Desonerado, models.DO_NOTHING, db_column='desonerado')
    cst = models.CharField(max_length=4)
    motivo = models.IntegerField()
    desconta_total = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'desonerado_configuracao'


class DetalhesFormacaoPrecoVenda(models.Model):
    codigo = models.AutoField(primary_key=True)
    cabecalho = models.ForeignKey(CabecalhoDetalhesFormacaoPrecoVenda, models.DO_NOTHING, db_column='cabecalho')
    indice = models.IntegerField()
    descricao = models.CharField(max_length=50)
    valor = models.DecimalField(max_digits=15, decimal_places=2)
    formula = models.CharField(max_length=255)
    negrito = models.BooleanField()
    italico = models.BooleanField()
    sublinhado = models.BooleanField()
    nomefonte = models.CharField(max_length=50)
    tamanhofonte = models.IntegerField()
    corfonte = models.CharField(max_length=50)
    corfundo = models.CharField(max_length=50)
    atualizacusto = models.BooleanField()
    atualizavenda = models.BooleanField()
    atualizaminimo = models.BooleanField()
    observacao = models.CharField(max_length=10000, blank=True, null=True)
    tipocampo = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'detalhes_formacao_preco_venda'


class DetalhesFormacaoPrecoVendaLog(models.Model):
    codigo = models.IntegerField(primary_key=True)
    cabecalho = models.IntegerField(blank=True, null=True)
    indice = models.IntegerField()
    descricao = models.CharField(max_length=-1)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535)
    formula = models.CharField(max_length=-1)
    negrito = models.BooleanField()
    italico = models.BooleanField()
    sublinhado = models.BooleanField()
    nomefonte = models.CharField(max_length=-1)
    tamanhofonte = models.IntegerField()
    corfonte = models.CharField(max_length=-1)
    corfundo = models.CharField(max_length=-1)
    atualizacusto = models.BooleanField()
    atualizavenda = models.BooleanField()
    atualizaminimo = models.BooleanField()
    observacao = models.CharField(max_length=-1, blank=True, null=True)
    tipocampo = models.CharField(max_length=-1, blank=True, null=True)
    datahora = models.DateTimeField()
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')

    class Meta:
        managed = False
        db_table = 'detalhes_formacao_preco_venda_log'


class Direito1(models.Model):
    opcao = models.IntegerField(primary_key=True)
    nivel = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'direito1'
        unique_together = (('opcao', 'usuario'),)


class DjangoAdminLog(models.Model):
    action_time = models.DateTimeField()
    object_id = models.TextField(blank=True, null=True)
    object_repr = models.CharField(max_length=200)
    action_flag = models.SmallIntegerField()
    change_message = models.TextField()
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING, blank=True, null=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'django_admin_log'


class DjangoContentType(models.Model):
    app_label = models.CharField(max_length=100)
    model = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'django_content_type'
        unique_together = (('app_label', 'model'),)


class DjangoMigrations(models.Model):
    app = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    applied = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_migrations'


class DjangoSession(models.Model):
    session_key = models.CharField(primary_key=True, max_length=40)
    session_data = models.TextField()
    expire_date = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_session'


class DocumentosGerenciais(models.Model):
    codigo = models.AutoField(primary_key=True)
    pedido = models.ForeignKey(CabecalhoOrdemServico, models.DO_NOTHING, db_column='pedido', blank=True, null=True)
    ccf = models.IntegerField(blank=True, null=True)
    gnf = models.IntegerField(blank=True, null=True)
    grg = models.IntegerField(blank=True, null=True)
    cdc = models.IntegerField(blank=True, null=True)
    denominacao = models.CharField(max_length=2, blank=True, null=True)
    forma_pgto = models.CharField(max_length=15, blank=True, null=True)
    data_emissao = models.DateField()
    hora_emissao = models.TimeField()
    numero_serie = models.CharField(max_length=-1, blank=True, null=True)
    cancelado = models.BooleanField(blank=True, null=True)
    coo = models.IntegerField(blank=True, null=True)
    exportou = models.BooleanField(blank=True, null=True)
    md5 = models.CharField(max_length=-1)
    troco = models.DecimalField(max_digits=15, decimal_places=2)
    terminal = models.ForeignKey('Terminais', models.DO_NOTHING, db_column='terminal')
    md5_h2 = models.CharField(max_length=-1)
    titulo_capitalizacao = models.IntegerField()
    cliente_doacao = models.IntegerField(blank=True, null=True)
    orcamento = models.ForeignKey(CabecalhoOrcamento, models.DO_NOTHING, db_column='orcamento', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'documentos_gerenciais'


class DreConfiguracoes(models.Model):
    codigo = models.AutoField(primary_key=True)
    descricao = models.CharField(max_length=50, blank=True, null=True)
    nivel = models.IntegerField(blank=True, null=True)
    local_consulta = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'dre_configuracoes'


class DreConfiguracoesHistoricos(models.Model):
    codigo = models.AutoField(primary_key=True)
    historico = models.ForeignKey('Historicos', models.DO_NOTHING, db_column='historico')
    tipo = models.ForeignKey(DreConfiguracoes, models.DO_NOTHING, db_column='tipo')

    class Meta:
        managed = False
        db_table = 'dre_configuracoes_historicos'


class DuplicatasEmitidas(models.Model):
    codigo = models.AutoField(primary_key=True)
    cod_parcela = models.ForeignKey('Parcelas', models.DO_NOTHING, db_column='cod_parcela', blank=True, null=True)
    valor = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)
    data = models.DateField()
    hora = models.TimeField()

    class Meta:
        managed = False
        db_table = 'duplicatas_emitidas'


class Embarque(models.Model):
    codigo = models.AutoField(primary_key=True)
    item_reserva = models.OneToOneField('ItensOrcamento', models.DO_NOTHING, db_column='item_reserva', blank=True, null=True)
    item_venda = models.OneToOneField('ItensOrdemServico', models.DO_NOTHING, db_column='item_venda', blank=True, null=True)
    dias_para_embarque = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'embarque'


class EmitindoNota(models.Model):
    pid = models.BigIntegerField(primary_key=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    data_operacao = models.DateTimeField()
    tipo_emissao = models.CharField(max_length=20)
    ambiente = models.IntegerField()
    serie = models.CharField(max_length=3)
    modelo = models.CharField(max_length=2)

    class Meta:
        managed = False
        db_table = 'emitindo_nota'
        unique_together = (('tipo_emissao', 'ambiente', 'serie', 'modelo'),)


class EnderecoCobranca(models.Model):
    codigo = models.AutoField(primary_key=True)
    endereco = models.CharField(max_length=50, blank=True, null=True)
    bairro = models.CharField(max_length=30, blank=True, null=True)
    numero = models.CharField(max_length=10, blank=True, null=True)
    cep = models.ForeignKey(Cidades, models.DO_NOTHING, db_column='cep', blank=True, null=True)
    cxpostal = models.CharField(max_length=10, blank=True, null=True)
    complemento = models.CharField(max_length=50, blank=True, null=True)
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente', blank=True, null=True)
    contato_ligar = models.CharField(max_length=50, blank=True, null=True)
    contato_atender = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'endereco_cobranca'


class EnderecoEntregaVenda(models.Model):
    codigo = models.AutoField(primary_key=True)
    ordem = models.ForeignKey(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', blank=True, null=True)
    endereco = models.CharField(max_length=100, blank=True, null=True)
    bairro = models.CharField(max_length=50, blank=True, null=True)
    cep = models.ForeignKey(Cidades, models.DO_NOTHING, db_column='cep', blank=True, null=True)
    fone = models.CharField(max_length=14, blank=True, null=True)
    ramal = models.CharField(max_length=5, blank=True, null=True)
    complemento = models.CharField(max_length=50, blank=True, null=True)
    reserva = models.ForeignKey(CabecalhoOrcamento, models.DO_NOTHING, db_column='reserva', blank=True, null=True)
    entrega = models.ForeignKey(CabecalhoEntrega, models.DO_NOTHING, db_column='entrega', blank=True, null=True)
    rota = models.ForeignKey('Rotas', models.DO_NOTHING, db_column='rota', blank=True, null=True)
    dataentrega = models.DateField(blank=True, null=True)
    horaentrega = models.TimeField(blank=True, null=True)
    contato = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'endereco_entrega_venda'


class EntradaSaidaManual(models.Model):
    codigo = models.AutoField(primary_key=True)
    tipo = models.CharField(max_length=1, blank=True, null=True)
    cd_produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='cd_produto', blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    quantidade = models.DecimalField(max_digits=15, decimal_places=3, blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)
    motivo = models.CharField(max_length=100, blank=True, null=True)
    preco_custo = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    preco_venda = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    nmotivo = models.ForeignKey('Motivos', models.DO_NOTHING, db_column='nmotivo', blank=True, null=True)
    hora_lancamento = models.TimeField()
    utiliza_lote = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'entrada_saida_manual'


class EntregaFuturaCompra(models.Model):
    item_compra = models.ForeignKey('ItensNotaCompra', models.DO_NOTHING, db_column='item_compra')
    item_pedido = models.ForeignKey('ItensCompra', models.DO_NOTHING, db_column='item_pedido')
    codigo = models.AutoField(primary_key=True)
    quantia = models.DecimalField(max_digits=15, decimal_places=5)

    class Meta:
        managed = False
        db_table = 'entrega_futura_compra'


class EnviaAberto(models.Model):
    pid = models.DecimalField(primary_key=True, max_digits=65535, decimal_places=65535)

    class Meta:
        managed = False
        db_table = 'envia_aberto'


class EquipamentosPos(models.Model):
    codigo = models.AutoField(primary_key=True)
    descricao = models.CharField(max_length=50)
    serial = models.CharField(max_length=50)
    adquirente = models.ForeignKey('Fornecedores', models.DO_NOTHING, db_column='adquirente', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'equipamentos_pos'


class Escalas(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nome = models.CharField(max_length=30, blank=True, null=True)
    segundaent1 = models.TimeField(blank=True, null=True)
    segundasai1 = models.TimeField(blank=True, null=True)
    segundaent2 = models.TimeField(blank=True, null=True)
    segundasai2 = models.TimeField(blank=True, null=True)
    tercaent1 = models.TimeField(blank=True, null=True)
    tercasai1 = models.TimeField(blank=True, null=True)
    tercaent2 = models.TimeField(blank=True, null=True)
    tercasai2 = models.TimeField(blank=True, null=True)
    quartaent1 = models.TimeField(blank=True, null=True)
    quartasai1 = models.TimeField(blank=True, null=True)
    quartaent2 = models.TimeField(blank=True, null=True)
    quartasai2 = models.TimeField(blank=True, null=True)
    quintaent1 = models.TimeField(blank=True, null=True)
    quintasai1 = models.TimeField(blank=True, null=True)
    quintaent2 = models.TimeField(blank=True, null=True)
    quintasai2 = models.TimeField(blank=True, null=True)
    sextaent1 = models.TimeField(blank=True, null=True)
    sextasai1 = models.TimeField(blank=True, null=True)
    sextaent2 = models.TimeField(blank=True, null=True)
    sextasai2 = models.TimeField(blank=True, null=True)
    sabadoent1 = models.TimeField(blank=True, null=True)
    sabadosai1 = models.TimeField(blank=True, null=True)
    sabadoent2 = models.TimeField(blank=True, null=True)
    sabadosai2 = models.TimeField(blank=True, null=True)
    domingoent1 = models.TimeField(blank=True, null=True)
    domingosai1 = models.TimeField(blank=True, null=True)
    domingoent2 = models.TimeField(blank=True, null=True)
    domingosai2 = models.TimeField(blank=True, null=True)
    tolerancia = models.TimeField(blank=True, null=True)
    dsr = models.CharField(max_length=1, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'escalas'


class EspecieParcela(models.Model):
    parcela = models.ForeignKey('ParcelasQuitadas', models.DO_NOTHING, db_column='parcela', blank=True, null=True)
    especie = models.ForeignKey('Especies', models.DO_NOTHING, db_column='especie', blank=True, null=True)
    valor = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'especie_parcela'


class Especies(models.Model):
    codigo = models.IntegerField(primary_key=True)
    descricao = models.CharField(max_length=50, blank=True, null=True)
    tipo = models.IntegerField(blank=True, null=True)
    tef = models.BooleanField(blank=True, null=True)
    confissao = models.BooleanField(blank=True, null=True)
    usa_programa_externo = models.BooleanField(blank=True, null=True)
    caminho_programa = models.CharField(max_length=55, blank=True, null=True)
    analisalimitecredito = models.BooleanField()
    utiliza_pos = models.BooleanField()
    indexador = models.DecimalField(max_digits=65535, decimal_places=65535)

    class Meta:
        managed = False
        db_table = 'especies'


class EspeciesEmitirNota(models.Model):
    codigo = models.BigIntegerField(blank=True, null=True)
    especie = models.IntegerField()
    valor = models.DecimalField(max_digits=65535, decimal_places=65535)
    valortotal = models.DecimalField(max_digits=65535, decimal_places=65535)
    saque = models.DecimalField(max_digits=65535, decimal_places=65535)
    pid = models.BigIntegerField()

    class Meta:
        managed = False
        db_table = 'especies_emitir_nota'


class Estados(models.Model):
    uf = models.CharField(primary_key=True, max_length=2)
    descricao = models.CharField(max_length=20, blank=True, null=True)
    aliquota_contrib = models.DecimalField(max_digits=5, decimal_places=2, blank=True, null=True)
    aliquota_ncontrib = models.DecimalField(max_digits=5, decimal_places=2, blank=True, null=True)
    intra_contrib = models.DecimalField(max_digits=65535, decimal_places=65535)
    intra_ncontrib = models.DecimalField(max_digits=65535, decimal_places=65535)
    inscricao_substituto = models.CharField(max_length=15, blank=True, null=True)
    autoriza_descontar_icms = models.BooleanField(blank=True, null=True)
    codigo_estado = models.IntegerField(blank=True, null=True)
    perfil = models.CharField(max_length=2)
    tipo_partilha = models.IntegerField()
    url_sintegra = models.CharField(max_length=255, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'estados'


class EstoqueInicial(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='cd_produto', blank=True, null=True)
    est_minimo = models.DecimalField(max_digits=15, decimal_places=3, blank=True, null=True)
    est_maximo = models.DecimalField(max_digits=15, decimal_places=3, blank=True, null=True)
    quantidade = models.DecimalField(max_digits=15, decimal_places=3, blank=True, null=True)
    data = models.DateField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'estoque_inicial'


class EtiqTemp(models.Model):
    produto = models.IntegerField(blank=True, null=True)
    descricao = models.CharField(max_length=60, blank=True, null=True)
    quantidade = models.IntegerField(blank=True, null=True)
    preco = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    referencia = models.CharField(max_length=20, blank=True, null=True)
    custo = models.CharField(max_length=15, blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    serial = models.CharField(max_length=20, blank=True, null=True)
    fornecedor = models.CharField(max_length=50, blank=True, null=True)
    codigo = models.AutoField(primary_key=True)
    item_nota_compra = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'etiq_temp'


class Etiquetas(models.Model):
    codigo = models.AutoField(primary_key=True)
    modelo = models.CharField(max_length=50, blank=True, null=True)
    margemsuperior = models.IntegerField(blank=True, null=True)
    margemesquerda = models.IntegerField(blank=True, null=True)
    espvertical = models.IntegerField(blank=True, null=True)
    esphorizontal = models.IntegerField(blank=True, null=True)
    linhas = models.IntegerField(blank=True, null=True)
    colunas = models.IntegerField(blank=True, null=True)
    larguraetiqueta = models.IntegerField(blank=True, null=True)
    alturaetiqueta = models.IntegerField(blank=True, null=True)
    tamanho = models.IntegerField(blank=True, null=True)
    espacamento = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'etiquetas'


class EtiquetasSequencia(models.Model):
    codigo = models.AutoField(primary_key=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    coluna = models.IntegerField(blank=True, null=True)
    linha = models.IntegerField(blank=True, null=True)
    desc_linha = models.CharField(max_length=-1, blank=True, null=True)
    desc_coluna = models.CharField(max_length=-1, blank=True, null=True)
    disponivel = models.BooleanField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'etiquetas_sequencia'


class EtiquetasVolumes(models.Model):
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    sequencial = models.IntegerField(blank=True, null=True)
    volume = models.CharField(max_length=-1, blank=True, null=True)
    data = models.DateField()
    hora = models.TimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'etiquetas_volumes'


class Eventosnfe(models.Model):
    codigo = models.AutoField(primary_key=True)
    chave = models.CharField(max_length=-1, blank=True, null=True)
    evento = models.IntegerField(blank=True, null=True)
    descricao_evento = models.CharField(max_length=-1, blank=True, null=True)
    status = models.IntegerField(blank=True, null=True)
    data = models.DateTimeField()
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador', blank=True, null=True)
    xml = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'eventosnfe'


class Extra1(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nome = models.CharField(max_length=100, blank=True, null=True)
    extra1 = models.CharField(max_length=56, blank=True, null=True)
    extra2 = models.CharField(max_length=56, blank=True, null=True)
    extra3 = models.CharField(max_length=56, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'extra1'


class Extra2(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nome = models.CharField(max_length=100, blank=True, null=True)
    extra1 = models.CharField(max_length=56, blank=True, null=True)
    extra2 = models.CharField(max_length=56, blank=True, null=True)
    extra3 = models.CharField(max_length=56, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'extra2'


class Faturas(models.Model):
    numero = models.AutoField(primary_key=True)
    valor = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    acrescimo = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    desconto = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    observacoes = models.CharField(max_length=100, blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)
    cd_observacao = models.IntegerField(blank=True, null=True)
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'faturas'


class FaturasApagar(models.Model):
    numero = models.AutoField(primary_key=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    acrescimo = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    desconto = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)
    fornecedor = models.ForeignKey('Fornecedores', models.DO_NOTHING, db_column='fornecedor', blank=True, null=True)
    data_lancamento = models.DateField(blank=True, null=True)
    hora_lancamento = models.TimeField(blank=True, null=True)
    observacao = models.ForeignKey('ObservacoesMovimentos', models.DO_NOTHING, db_column='observacao', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'faturas_apagar'


class FaturasApagarDestino(models.Model):
    fatura = models.OneToOneField(FaturasApagar, models.DO_NOTHING, db_column='fatura', primary_key=True)
    apagar = models.ForeignKey(Apagar, models.DO_NOTHING, db_column='apagar')

    class Meta:
        managed = False
        db_table = 'faturas_apagar_destino'
        unique_together = (('fatura', 'apagar'),)


class FaturasApagarOrigem(models.Model):
    fatura = models.OneToOneField(FaturasApagar, models.DO_NOTHING, db_column='fatura', primary_key=True)
    apagar = models.ForeignKey(Apagar, models.DO_NOTHING, db_column='apagar')

    class Meta:
        managed = False
        db_table = 'faturas_apagar_origem'
        unique_together = (('fatura', 'apagar'),)


class FaturasDestino(models.Model):
    fatura = models.OneToOneField(Faturas, models.DO_NOTHING, db_column='fatura', primary_key=True)
    parcela = models.ForeignKey('Parcelas', models.DO_NOTHING, db_column='parcela')

    class Meta:
        managed = False
        db_table = 'faturas_destino'
        unique_together = (('fatura', 'parcela'),)


class FaturasOrigem(models.Model):
    fatura = models.OneToOneField(Faturas, models.DO_NOTHING, db_column='fatura', primary_key=True)
    parcela = models.ForeignKey('Parcelas', models.DO_NOTHING, db_column='parcela')

    class Meta:
        managed = False
        db_table = 'faturas_origem'
        unique_together = (('fatura', 'parcela'),)


class FaturasResponsavel(models.Model):
    fatura = models.OneToOneField(Faturas, models.DO_NOTHING, db_column='fatura', primary_key=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'faturas_responsavel'


class Favoritos(models.Model):
    codigo = models.AutoField(primary_key=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    nome = models.CharField(max_length=-1)
    descricao = models.CharField(max_length=-1)
    caption = models.CharField(max_length=13)
    atalho = models.CharField(max_length=-1, blank=True, null=True)
    imageindex = models.IntegerField()
    sequencia = models.IntegerField()
    posicao = models.IntegerField()
    tiposistema = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'favoritos'
        unique_together = (('usuario', 'nome'),)


class Filiais(models.Model):
    codigo = models.AutoField(primary_key=True)
    nome = models.CharField(max_length=50, blank=True, null=True)
    base = models.CharField(max_length=20, blank=True, null=True)
    usuario = models.CharField(max_length=20, blank=True, null=True)
    senha = models.CharField(max_length=20, blank=True, null=True)
    ip = models.CharField(max_length=20, blank=True, null=True)
    cnpj = models.CharField(max_length=18, blank=True, null=True)
    online = models.BooleanField(blank=True, null=True)
    ie = models.CharField(max_length=20, blank=True, null=True)
    endereco = models.CharField(max_length=50, blank=True, null=True)
    bairro = models.CharField(max_length=50, blank=True, null=True)
    fone = models.CharField(max_length=14, blank=True, null=True)
    cep = models.ForeignKey(Cidades, models.DO_NOTHING, db_column='cep', blank=True, null=True)
    porta = models.IntegerField()
    matriz = models.BooleanField(blank=True, null=True)
    marca = models.CharField(max_length=1, blank=True, null=True)
    email = models.CharField(max_length=50, blank=True, null=True)
    tipo_comercio = models.IntegerField(blank=True, null=True)
    desonerado = models.ForeignKey(Desonerado, models.DO_NOTHING, db_column='desonerado', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'filiais'


class FiliaisTransferir(models.Model):
    origem = models.ForeignKey(Filiais, models.DO_NOTHING, db_column='origem')
    destino = models.ForeignKey(Filiais, models.DO_NOTHING, db_column='destino')
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'filiais_transferir'


class ForeignSchemas(models.Model):
    codigo = models.AutoField(primary_key=True)
    nome_schema = models.CharField(max_length=100, blank=True, null=True)
    nome_tabela = models.CharField(max_length=100, blank=True, null=True)
    nome_foreign = models.CharField(max_length=100, blank=True, null=True)
    sql = models.CharField(max_length=1000, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'foreign_schemas'


class FormacaoPreco(models.Model):
    codigo = models.AutoField(primary_key=True)
    descricao = models.CharField(max_length=50, blank=True, null=True)
    percentual = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'formacao_preco'


class Fornecedores(models.Model):
    codigo = models.IntegerField(primary_key=True)
    cnpj = models.CharField(max_length=18, blank=True, null=True)
    inscricaoest = models.CharField(max_length=20, blank=True, null=True)
    datacad = models.DateField()
    nome = models.CharField(max_length=50, blank=True, null=True)
    fantasia = models.CharField(max_length=50, blank=True, null=True)
    endereco = models.CharField(max_length=50, blank=True, null=True)
    bairro = models.CharField(max_length=50, blank=True, null=True)
    cep = models.ForeignKey(Cidades, models.DO_NOTHING, db_column='cep', blank=True, null=True)
    referencia = models.CharField(max_length=50, blank=True, null=True)
    fone = models.CharField(max_length=14, blank=True, null=True)
    ramal = models.CharField(max_length=6, blank=True, null=True)
    fax = models.CharField(max_length=14, blank=True, null=True)
    contato = models.CharField(max_length=50, blank=True, null=True)
    email = models.CharField(max_length=50, blank=True, null=True)
    url = models.CharField(max_length=50, blank=True, null=True)
    representante = models.CharField(max_length=50, blank=True, null=True)
    fonerepresentante = models.CharField(max_length=14, blank=True, null=True)
    celularrepresentante = models.CharField(max_length=14, blank=True, null=True)
    atividade = models.ForeignKey(Atividades, models.DO_NOTHING, db_column='atividade', blank=True, null=True)
    observacoes = models.CharField(max_length=340, blank=True, null=True)
    cxpostal = models.CharField(max_length=15, blank=True, null=True)
    codobservacao = models.ForeignKey('Observacoes', models.DO_NOTHING, db_column='codobservacao', blank=True, null=True)
    fone2 = models.CharField(max_length=20, blank=True, null=True)
    produtorrural = models.BooleanField(blank=True, null=True)
    codigo_pais = models.IntegerField(blank=True, null=True)
    dias_atualizar = models.IntegerField(blank=True, null=True)
    industria = models.BooleanField()
    numero = models.CharField(max_length=10, blank=True, null=True)
    complemento = models.CharField(max_length=50, blank=True, null=True)
    regime_especial = models.BooleanField()
    simples = models.BooleanField()
    permite_duplicar_cpf_cnpj = models.BooleanField()
    tipo_pessoa = models.IntegerField()
    email_xml = models.CharField(max_length=50, blank=True, null=True)
    id_estrangeiro = models.CharField(max_length=20, blank=True, null=True)
    chave_requisicao_mfe = models.CharField(max_length=50)
    desonerado = models.ForeignKey(Desonerado, models.DO_NOTHING, db_column='desonerado', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'fornecedores'


class FornecedoresAlteracoesSped(models.Model):
    codigo = models.AutoField(primary_key=True)
    fornecedor = models.ForeignKey(Fornecedores, models.DO_NOTHING, db_column='fornecedor', blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    campo = models.IntegerField(blank=True, null=True)
    antigo = models.CharField(max_length=-1, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'fornecedores_alteracoes_sped'


class FornecedoresAlvara(models.Model):
    fornecedor = models.OneToOneField(Fornecedores, models.DO_NOTHING, db_column='fornecedor', primary_key=True)
    numero_alvara = models.CharField(max_length=20, blank=True, null=True)
    certidao = models.CharField(max_length=20, blank=True, null=True)
    data_ven_alvara = models.DateField(blank=True, null=True)
    responsavel = models.CharField(max_length=50, blank=True, null=True)
    registro_responsavel = models.CharField(max_length=20, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'fornecedores_alvara'


class Fracionadores(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nome = models.CharField(max_length=100, blank=True, null=True)
    fracionador1 = models.CharField(max_length=56, blank=True, null=True)
    fracionador2 = models.CharField(max_length=56, blank=True, null=True)
    fracionador3 = models.CharField(max_length=56, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'fracionadores'


class FundosSistema(models.Model):
    codigo = models.AutoField(primary_key=True)
    sistema = models.CharField(max_length=-1)
    nome = models.CharField(max_length=-1)
    imagem = models.BinaryField()

    class Meta:
        managed = False
        db_table = 'fundos_sistema'


class GeneroItem(models.Model):
    codigo = models.AutoField(primary_key=True)
    indice = models.CharField(max_length=3, blank=True, null=True)
    descricao = models.CharField(max_length=1000, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'genero_item'


class GertecBalancas(models.Model):
    codigo = models.AutoField(primary_key=True)
    serial = models.BigIntegerField(blank=True, null=True)
    porta_com = models.IntegerField(blank=True, null=True)
    usa_balanca = models.BooleanField()
    baud_rate = models.IntegerField(blank=True, null=True)
    bits = models.IntegerField(blank=True, null=True)
    parity = models.IntegerField(blank=True, null=True)
    stop = models.IntegerField(blank=True, null=True)
    handshake = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'gertec_balancas'


class GradeProdutos(models.Model):
    codigo = models.AutoField(primary_key=True)
    nome = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'grade_produtos'


class GradeProdutosAvulsa(models.Model):
    codigo = models.AutoField(primary_key=True)
    grade = models.ForeignKey(GradeProdutos, models.DO_NOTHING, db_column='grade', blank=True, null=True)
    linha = models.IntegerField(blank=True, null=True)
    coluna = models.IntegerField(blank=True, null=True)
    cd_item = models.ForeignKey('ItensNotaAvulsa', models.DO_NOTHING, db_column='cd_item', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'grade_produtos_avulsa'


class GradeProdutosCompra(models.Model):
    codigo = models.AutoField(primary_key=True)
    grade = models.ForeignKey(GradeProdutos, models.DO_NOTHING, db_column='grade', blank=True, null=True)
    linha = models.IntegerField(blank=True, null=True)
    coluna = models.IntegerField(blank=True, null=True)
    cd_item = models.ForeignKey('ItensNotaCompra', models.DO_NOTHING, db_column='cd_item', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'grade_produtos_compra'


class GradeProdutosConferenciaEstoque(models.Model):
    codigo = models.AutoField(primary_key=True)
    grade = models.ForeignKey(GradeProdutos, models.DO_NOTHING, db_column='grade', blank=True, null=True)
    linha = models.IntegerField(blank=True, null=True)
    coluna = models.IntegerField(blank=True, null=True)
    cd_item = models.ForeignKey('ItensConferenciaEstoque', models.DO_NOTHING, db_column='cd_item', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'grade_produtos_conferencia_estoque'


class GradeProdutosCustosVenda(models.Model):
    codigo = models.AutoField(primary_key=True)
    grade = models.ForeignKey(GradeProdutos, models.DO_NOTHING, db_column='grade', blank=True, null=True)
    linha = models.IntegerField(blank=True, null=True)
    coluna = models.IntegerField(blank=True, null=True)
    cd_item = models.ForeignKey(CabecalhoOrdemServicoDespesas, models.DO_NOTHING, db_column='cd_item', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'grade_produtos_custos_venda'


class GradeProdutosDados(models.Model):
    codigo = models.AutoField(primary_key=True)
    grade = models.ForeignKey(GradeProdutos, models.DO_NOTHING, db_column='grade', blank=True, null=True)
    linha = models.IntegerField(blank=True, null=True)
    coluna = models.IntegerField(blank=True, null=True)
    descricao = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'grade_produtos_dados'


class GradeProdutosDevolucao(models.Model):
    codigo = models.AutoField(primary_key=True)
    grade = models.ForeignKey(GradeProdutos, models.DO_NOTHING, db_column='grade', blank=True, null=True)
    linha = models.IntegerField(blank=True, null=True)
    coluna = models.IntegerField(blank=True, null=True)
    cd_item = models.ForeignKey('ItensDevolucao', models.DO_NOTHING, db_column='cd_item', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'grade_produtos_devolucao'


class GradeProdutosDevolviEmprestado(models.Model):
    codigo = models.AutoField(primary_key=True)
    grade = models.ForeignKey(GradeProdutos, models.DO_NOTHING, db_column='grade', blank=True, null=True)
    linha = models.IntegerField(blank=True, null=True)
    coluna = models.IntegerField(blank=True, null=True)
    cd_item = models.ForeignKey('ProdutosEmprestadosDevolvidos', models.DO_NOTHING, db_column='cd_item', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'grade_produtos_devolvi_emprestado'


class GradeProdutosDevolviEmprestei(models.Model):
    codigo = models.AutoField(primary_key=True)
    grade = models.ForeignKey(GradeProdutos, models.DO_NOTHING, db_column='grade', blank=True, null=True)
    linha = models.IntegerField(blank=True, null=True)
    coluna = models.IntegerField(blank=True, null=True)
    cd_item = models.ForeignKey('ProdutosEmpresteiDevolvi', models.DO_NOTHING, db_column='cd_item', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'grade_produtos_devolvi_emprestei'


class GradeProdutosEmprestado(models.Model):
    codigo = models.AutoField(primary_key=True)
    grade = models.ForeignKey(GradeProdutos, models.DO_NOTHING, db_column='grade', blank=True, null=True)
    linha = models.IntegerField(blank=True, null=True)
    coluna = models.IntegerField(blank=True, null=True)
    cd_item = models.ForeignKey('ProdutosEmprestados', models.DO_NOTHING, db_column='cd_item', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'grade_produtos_emprestado'


class GradeProdutosEmprestei(models.Model):
    codigo = models.AutoField(primary_key=True)
    grade = models.ForeignKey(GradeProdutos, models.DO_NOTHING, db_column='grade', blank=True, null=True)
    linha = models.IntegerField(blank=True, null=True)
    coluna = models.IntegerField(blank=True, null=True)
    cd_item = models.ForeignKey('ProdutosEmprestei', models.DO_NOTHING, db_column='cd_item', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'grade_produtos_emprestei'


class GradeProdutosEntradaSaidaManual(models.Model):
    codigo = models.AutoField(primary_key=True)
    grade = models.ForeignKey(GradeProdutos, models.DO_NOTHING, db_column='grade', blank=True, null=True)
    linha = models.IntegerField(blank=True, null=True)
    coluna = models.IntegerField(blank=True, null=True)
    cd_item = models.ForeignKey(EntradaSaidaManual, models.DO_NOTHING, db_column='cd_item', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'grade_produtos_entrada_saida_manual'


class GradeProdutosInicio(models.Model):
    codigo = models.AutoField(primary_key=True)
    grade = models.ForeignKey(GradeProdutos, models.DO_NOTHING, db_column='grade', blank=True, null=True)
    linha = models.IntegerField(blank=True, null=True)
    coluna = models.IntegerField(blank=True, null=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'grade_produtos_inicio'


class GradeProdutosItensEntrega(models.Model):
    codigo = models.AutoField(primary_key=True)
    grade = models.ForeignKey(GradeProdutos, models.DO_NOTHING, db_column='grade', blank=True, null=True)
    linha = models.IntegerField(blank=True, null=True)
    coluna = models.IntegerField(blank=True, null=True)
    cd_item = models.ForeignKey('ItensEntrega', models.DO_NOTHING, db_column='cd_item', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'grade_produtos_itens_entrega'


class GradeProdutosLog(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_item = models.IntegerField(blank=True, null=True)
    grade = models.ForeignKey(GradeProdutos, models.DO_NOTHING, db_column='grade', blank=True, null=True)
    linha = models.IntegerField(blank=True, null=True)
    coluna = models.IntegerField(blank=True, null=True)
    entrada = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    saida = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    data = models.DateField()
    hora = models.TimeField()
    operacao = models.CharField(max_length=-1, blank=True, null=True)
    desc_linha = models.CharField(max_length=-1, blank=True, null=True)
    desc_coluna = models.CharField(max_length=-1, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'grade_produtos_log'


class GradeProdutosOrdemServico(models.Model):
    codigo = models.AutoField(primary_key=True)
    grade = models.ForeignKey(GradeProdutos, models.DO_NOTHING, db_column='grade', blank=True, null=True)
    linha = models.IntegerField(blank=True, null=True)
    coluna = models.IntegerField(blank=True, null=True)
    cd_item = models.ForeignKey('OsServicos', models.DO_NOTHING, db_column='cd_item', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'grade_produtos_ordem_servico'


class GradeProdutosProducao(models.Model):
    codigo = models.AutoField(primary_key=True)
    grade = models.ForeignKey(GradeProdutos, models.DO_NOTHING, db_column='grade', blank=True, null=True)
    linha = models.IntegerField(blank=True, null=True)
    coluna = models.IntegerField(blank=True, null=True)
    cd_item = models.ForeignKey('ItensProducao', models.DO_NOTHING, db_column='cd_item', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'grade_produtos_producao'


class GradeProdutosProducaoFormula(models.Model):
    codigo = models.AutoField(primary_key=True)
    grade = models.ForeignKey(GradeProdutos, models.DO_NOTHING, db_column='grade', blank=True, null=True)
    linha = models.IntegerField(blank=True, null=True)
    coluna = models.IntegerField(blank=True, null=True)
    cd_item = models.ForeignKey('ItensFormula', models.DO_NOTHING, db_column='cd_item', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'grade_produtos_producao_formula'


class GradeProdutosProducaoPerda(models.Model):
    codigo = models.AutoField(primary_key=True)
    grade = models.ForeignKey(GradeProdutos, models.DO_NOTHING, db_column='grade', blank=True, null=True)
    linha = models.IntegerField(blank=True, null=True)
    coluna = models.IntegerField(blank=True, null=True)
    cd_item = models.ForeignKey('ItensPerda', models.DO_NOTHING, db_column='cd_item', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'grade_produtos_producao_perda'


class GradeProdutosRemessaConsertoEntrada(models.Model):
    codigo = models.AutoField(primary_key=True)
    grade = models.ForeignKey(GradeProdutos, models.DO_NOTHING, db_column='grade', blank=True, null=True)
    linha = models.IntegerField(blank=True, null=True)
    coluna = models.IntegerField(blank=True, null=True)
    cd_item = models.ForeignKey('ItensRemessaEntrada', models.DO_NOTHING, db_column='cd_item', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'grade_produtos_remessa_conserto_entrada'


class GradeProdutosRemessaDevolucaoFornecedor(models.Model):
    codigo = models.AutoField(primary_key=True)
    grade = models.ForeignKey(GradeProdutos, models.DO_NOTHING, db_column='grade', blank=True, null=True)
    linha = models.IntegerField(blank=True, null=True)
    coluna = models.IntegerField(blank=True, null=True)
    cd_item = models.ForeignKey('ItensRemessa', models.DO_NOTHING, db_column='cd_item', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'grade_produtos_remessa_devolucao_fornecedor'


class GradeProdutosReserva(models.Model):
    codigo = models.AutoField(primary_key=True)
    grade = models.ForeignKey(GradeProdutos, models.DO_NOTHING, db_column='grade', blank=True, null=True)
    linha = models.IntegerField(blank=True, null=True)
    coluna = models.IntegerField(blank=True, null=True)
    cd_item = models.ForeignKey('ItensOrcamento', models.DO_NOTHING, db_column='cd_item', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'grade_produtos_reserva'


class GradeProdutosSaldo(models.Model):
    codigo = models.AutoField(primary_key=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    grade = models.ForeignKey(GradeProdutos, models.DO_NOTHING, db_column='grade', blank=True, null=True)
    linha = models.IntegerField(blank=True, null=True)
    coluna = models.IntegerField(blank=True, null=True)
    saldo = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'grade_produtos_saldo'


class GradeProdutosTransferencia(models.Model):
    codigo = models.AutoField(primary_key=True)
    grade = models.ForeignKey(GradeProdutos, models.DO_NOTHING, db_column='grade', blank=True, null=True)
    linha = models.IntegerField(blank=True, null=True)
    coluna = models.IntegerField(blank=True, null=True)
    cd_item = models.ForeignKey('ItensTransferencia', models.DO_NOTHING, db_column='cd_item', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'grade_produtos_transferencia'


class GradeProdutosTransferenciaEntrada(models.Model):
    codigo = models.AutoField(primary_key=True)
    grade = models.ForeignKey(GradeProdutos, models.DO_NOTHING, db_column='grade', blank=True, null=True)
    linha = models.IntegerField(blank=True, null=True)
    coluna = models.IntegerField(blank=True, null=True)
    cd_item = models.ForeignKey('ItensTransferenciaEntrada', models.DO_NOTHING, db_column='cd_item', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'grade_produtos_transferencia_entrada'


class GradeProdutosVenda(models.Model):
    codigo = models.AutoField(primary_key=True)
    grade = models.ForeignKey(GradeProdutos, models.DO_NOTHING, db_column='grade', blank=True, null=True)
    linha = models.IntegerField(blank=True, null=True)
    coluna = models.IntegerField(blank=True, null=True)
    cd_item = models.ForeignKey('ItensOrdemServico', models.DO_NOTHING, db_column='cd_item', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'grade_produtos_venda'


class GrupoClientes(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nome = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'grupo_clientes'


class GrupoProdutos(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nome = models.CharField(max_length=50, blank=True, null=True)
    comissao_vista = models.DecimalField(max_digits=10, decimal_places=3, blank=True, null=True)
    comissao_prazo = models.DecimalField(max_digits=10, decimal_places=3, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'grupo_produtos'


class GrupoProdutosMarcados(models.Model):
    grupo = models.OneToOneField(GrupoProdutos, models.DO_NOTHING, db_column='grupo', primary_key=True)

    class Meta:
        managed = False
        db_table = 'grupo_produtos_marcados'


class Historicos(models.Model):
    codigo = models.AutoField(primary_key=True)
    descricao = models.CharField(max_length=50, blank=True, null=True)
    operacao = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    calcula = models.CharField(max_length=1, blank=True, null=True)
    tipo = models.CharField(max_length=1, blank=True, null=True)
    nivel = models.CharField(max_length=25, blank=True, null=True)
    tipooperacao = models.CharField(max_length=1, blank=True, null=True)
    custo = models.ForeignKey(Centrocustos, models.DO_NOTHING, db_column='custo', blank=True, null=True)
    divisao = models.IntegerField(blank=True, null=True)
    marca = models.CharField(max_length=1, blank=True, null=True)
    ativo = models.BooleanField(blank=True, null=True)
    data_desativado = models.DateField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'historicos'


class Impostos(models.Model):
    codigo = models.AutoField(primary_key=True)
    item_ordem_servico = models.OneToOneField('ItensOrdemServico', models.DO_NOTHING, db_column='item_ordem_servico', blank=True, null=True)
    item_orcamento = models.OneToOneField('ItensOrcamento', models.DO_NOTHING, db_column='item_orcamento', blank=True, null=True)
    item_transferencia = models.OneToOneField('ItensTransferencia', models.DO_NOTHING, db_column='item_transferencia', blank=True, null=True)
    item_remessa_conserto = models.OneToOneField('ItensRemessa', models.DO_NOTHING, db_column='item_remessa_conserto', blank=True, null=True)
    item_devolucao_fornecedor = models.OneToOneField('ItensRemessa', models.DO_NOTHING, db_column='item_devolucao_fornecedor', blank=True, null=True)
    item_remessa_venda_fora = models.OneToOneField('ItensRemessaVendaFora', models.DO_NOTHING, db_column='item_remessa_venda_fora', blank=True, null=True)
    item_compra = models.ForeignKey('ItensNotaCompra', models.DO_NOTHING, db_column='item_compra', blank=True, null=True)
    item_avulsa = models.OneToOneField('ItensNotaAvulsa', models.DO_NOTHING, db_column='item_avulsa', blank=True, null=True)
    item_devolucao = models.OneToOneField('ItensDevolucao', models.DO_NOTHING, db_column='item_devolucao', blank=True, null=True)
    item_retorno_fora = models.OneToOneField('ItensRetornoVendaFora', models.DO_NOTHING, db_column='item_retorno_fora', blank=True, null=True)
    item_futura = models.OneToOneField('ItensEntrega', models.DO_NOTHING, db_column='item_futura', blank=True, null=True)
    simples = models.BooleanField()
    item_formula = models.ForeignKey('ItensFormula', models.DO_NOTHING, db_column='item_formula', blank=True, null=True)
    item_perda = models.ForeignKey('ItensPerda', models.DO_NOTHING, db_column='item_perda', blank=True, null=True)
    codigo_anp = models.CharField(max_length=9, blank=True, null=True)
    item_producao = models.ForeignKey('ItensProducao', models.DO_NOTHING, db_column='item_producao', blank=True, null=True)
    beneficio = models.CharField(max_length=10)
    destinatario_simples = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'impostos'


class ImpostosFederais(models.Model):
    cont_social = models.DecimalField(max_digits=65535, decimal_places=65535)
    ir = models.DecimalField(max_digits=65535, decimal_places=65535)
    data = models.DateField(blank=True, null=True)
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'impostos_federais'


class ImpostosIcms(models.Model):
    cst_venda = models.CharField(max_length=4, blank=True, null=True)
    base_icms = models.DecimalField(max_digits=15, decimal_places=2)
    reducao_icms = models.DecimalField(max_digits=15, decimal_places=2)
    base_reduzida = models.DecimalField(max_digits=15, decimal_places=2)
    aliquota_icms = models.DecimalField(max_digits=15, decimal_places=2)
    valor_icms = models.DecimalField(max_digits=15, decimal_places=2)
    tipo_reducao = models.IntegerField()
    cd_imposto = models.OneToOneField(Impostos, models.DO_NOTHING, db_column='cd_imposto', primary_key=True)
    fundo_combate_pobreza_percentual = models.DecimalField(max_digits=6, decimal_places=2)
    fundo_combate_pobreza_valor = models.DecimalField(max_digits=65535, decimal_places=65535)
    aliquota_interna = models.DecimalField(max_digits=65535, decimal_places=65535)
    partilha = models.DecimalField(max_digits=6, decimal_places=2)
    valor_partilhado_origem = models.DecimalField(max_digits=65535, decimal_places=65535)
    valor_partilhado_destino = models.DecimalField(max_digits=65535, decimal_places=65535)
    icms_aliquota_interestadual = models.DecimalField(max_digits=15, decimal_places=2)
    base_reduzida_destino = models.DecimalField(max_digits=15, decimal_places=2)
    tipo_partilha = models.IntegerField()
    add_fcp_valor_icms = models.BooleanField()
    valor_recuperacao = models.DecimalField(max_digits=15, decimal_places=2)
    valor_recuperacao_fcp = models.DecimalField(max_digits=15, decimal_places=2)
    percentual_credito_simples = models.DecimalField(max_digits=20, decimal_places=2)
    valor_credito_simples = models.DecimalField(max_digits=20, decimal_places=2)
    efetivo_base = models.DecimalField(max_digits=20, decimal_places=2)
    efetivo_reducao = models.DecimalField(max_digits=20, decimal_places=2)
    efetivo_base_reduzida = models.DecimalField(max_digits=20, decimal_places=2)
    efetivo_percentual = models.DecimalField(max_digits=20, decimal_places=2)
    efetivo_valor = models.DecimalField(max_digits=20, decimal_places=2)
    diferido_percentual = models.DecimalField(max_digits=20, decimal_places=2)
    diferido_valor = models.DecimalField(max_digits=20, decimal_places=2)
    valor_icms_original = models.DecimalField(max_digits=20, decimal_places=2)
    desonerado_percentual = models.DecimalField(max_digits=20, decimal_places=2)
    desonerado_valor = models.DecimalField(max_digits=20, decimal_places=2)
    desonerado_desconta_total = models.BooleanField()
    desonerado_motivo = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'impostos_icms'


class ImpostosIcmsSt(models.Model):
    base_icms_st = models.DecimalField(max_digits=15, decimal_places=2)
    reducao_base_icms_st = models.DecimalField(max_digits=15, decimal_places=2)
    base_reduzida = models.DecimalField(max_digits=15, decimal_places=2)
    aliquota_interna_icms = models.DecimalField(max_digits=15, decimal_places=2)
    aliquota_interestadual = models.DecimalField(max_digits=15, decimal_places=2)
    mva = models.DecimalField(max_digits=15, decimal_places=2)
    reducao_mva = models.DecimalField(max_digits=15, decimal_places=2)
    mva_reduzido = models.DecimalField(max_digits=15, decimal_places=2)
    valor_icms_st = models.DecimalField(max_digits=15, decimal_places=2)
    base_icms_st_retido = models.DecimalField(max_digits=15, decimal_places=2)
    valor_icms_st_retido = models.DecimalField(max_digits=15, decimal_places=2)
    lei = models.CharField(max_length=-1, blank=True, null=True)
    icmsbase = models.DecimalField(max_digits=15, decimal_places=2)
    icmsreducao = models.DecimalField(max_digits=15, decimal_places=2)
    icmsaliquota = models.DecimalField(max_digits=15, decimal_places=2)
    icmsvalor = models.DecimalField(max_digits=15, decimal_places=2)
    cd_imposto = models.OneToOneField(Impostos, models.DO_NOTHING, db_column='cd_imposto', primary_key=True)
    cest = models.CharField(max_length=10, blank=True, null=True)
    fundo_combate_pobreza_percentual = models.DecimalField(max_digits=6, decimal_places=2)
    fundo_combate_pobreza_valor = models.DecimalField(max_digits=65535, decimal_places=65535)
    add_fcp_valor_icms_st = models.BooleanField()
    fundo_combate_pobreza_percentual_retido = models.DecimalField(max_digits=6, decimal_places=2)
    fundo_combate_pobreza_valor_retido = models.DecimalField(max_digits=65535, decimal_places=65535)
    add_fcp_valor_icms_st_retido = models.BooleanField()
    percentual_st_retido = models.DecimalField(max_digits=65535, decimal_places=65535)
    cnpj_fabricante = models.CharField(max_length=18)
    valor_recuperacao_fcp = models.DecimalField(max_digits=15, decimal_places=2)
    valor_recuperacao = models.DecimalField(max_digits=15, decimal_places=2)
    valor_icms_substituto = models.DecimalField(max_digits=15, decimal_places=2)

    class Meta:
        managed = False
        db_table = 'impostos_icms_st'


class ImpostosIpi(models.Model):
    cst_ipi = models.CharField(max_length=2, blank=True, null=True)
    enquadramento_ipi = models.CharField(max_length=3, blank=True, null=True)
    base_ipi = models.DecimalField(max_digits=15, decimal_places=2)
    aliquota_ipi = models.DecimalField(max_digits=15, decimal_places=2)
    valor_ipi = models.DecimalField(max_digits=15, decimal_places=2)
    cd_imposto = models.OneToOneField(Impostos, models.DO_NOTHING, db_column='cd_imposto', primary_key=True)
    percentual_devolvido = models.DecimalField(max_digits=15, decimal_places=2)

    class Meta:
        managed = False
        db_table = 'impostos_ipi'


class ImpostosIss(models.Model):
    base_iss = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    aliquota_iss = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    valor_iss = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    irrf = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    valor_irrf = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    csll = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    valor_csll = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    inss = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    valor_inss = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    cd_imposto = models.OneToOneField(Impostos, models.DO_NOTHING, db_column='cd_imposto', primary_key=True)
    base_iss_retido = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    valor_iss_retido = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    aliquota_iss_retido = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    descricao_tributacao = models.CharField(max_length=-1)

    class Meta:
        managed = False
        db_table = 'impostos_iss'


class ImpostosPisCofins(models.Model):
    base_pis_cofins = models.DecimalField(max_digits=15, decimal_places=2)
    cst_pis = models.CharField(max_length=2, blank=True, null=True)
    cst_cofins = models.CharField(max_length=2, blank=True, null=True)
    aliquota_pis = models.DecimalField(max_digits=15, decimal_places=2)
    aliquota_cofins = models.DecimalField(max_digits=15, decimal_places=2)
    valor_pis = models.DecimalField(max_digits=15, decimal_places=2)
    valor_cofins = models.DecimalField(max_digits=15, decimal_places=2)
    cd_imposto = models.OneToOneField(Impostos, models.DO_NOTHING, db_column='cd_imposto', primary_key=True)

    class Meta:
        managed = False
        db_table = 'impostos_pis_cofins'


class Indicesconversao(models.Model):
    codigo = models.IntegerField(primary_key=True)
    data = models.DateField(blank=True, null=True)
    valor = models.DecimalField(max_digits=15, decimal_places=5, blank=True, null=True)
    coddescricao = models.ForeignKey(Especies, models.DO_NOTHING, db_column='coddescricao', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'indicesconversao'


class InfBancariaForn(models.Model):
    codigo = models.AutoField(primary_key=True)
    banco = models.CharField(max_length=30, blank=True, null=True)
    agencia = models.CharField(max_length=10, blank=True, null=True)
    conta = models.CharField(max_length=15, blank=True, null=True)
    titular = models.CharField(max_length=50, blank=True, null=True)
    fornecedor = models.ForeignKey(Fornecedores, models.DO_NOTHING, db_column='fornecedor', blank=True, null=True)
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente', blank=True, null=True)
    vendedor = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='vendedor', blank=True, null=True)
    cd_banco = models.ForeignKey(Bancos, models.DO_NOTHING, db_column='cd_banco', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'inf_bancaria_forn'


class InfBoletos(models.Model):
    cd_convenio = models.OneToOneField(ConvenioBancos, models.DO_NOTHING, db_column='cd_convenio', primary_key=True)
    aceite = models.IntegerField(blank=True, null=True)
    carteira = models.IntegerField(blank=True, null=True)
    cd_movimento = models.IntegerField(blank=True, null=True)
    especie = models.IntegerField(blank=True, null=True)
    emite_boleto = models.IntegerField(blank=True, null=True)
    distr_boleto = models.IntegerField(blank=True, null=True)
    protestar = models.IntegerField(blank=True, null=True)
    dia_protesto = models.IntegerField(blank=True, null=True)
    mensagem1 = models.CharField(max_length=40, blank=True, null=True)
    mensagem2 = models.CharField(max_length=40, blank=True, null=True)
    mensagem3 = models.CharField(max_length=40, blank=True, null=True)
    mensagem4 = models.CharField(max_length=40, blank=True, null=True)
    mensagem5 = models.CharField(max_length=40, blank=True, null=True)
    remessa = models.IntegerField(blank=True, null=True)
    tipo_documento = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'inf_boletos'


class InformacoesExtras(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nome = models.CharField(max_length=50, blank=True, null=True)
    extra1 = models.CharField(max_length=56, blank=True, null=True)
    extra2 = models.CharField(max_length=56, blank=True, null=True)
    extra3 = models.CharField(max_length=56, blank=True, null=True)
    extra4 = models.CharField(max_length=56, blank=True, null=True)
    extra5 = models.CharField(max_length=56, blank=True, null=True)
    extra6 = models.CharField(max_length=56, blank=True, null=True)
    extra7 = models.CharField(max_length=56, blank=True, null=True)
    extra8 = models.CharField(max_length=56, blank=True, null=True)
    extra9 = models.CharField(max_length=56, blank=True, null=True)
    extra10 = models.CharField(max_length=56, blank=True, null=True)
    extra11 = models.CharField(max_length=56, blank=True, null=True)
    extra12 = models.CharField(max_length=56, blank=True, null=True)
    extra13 = models.CharField(max_length=56, blank=True, null=True)
    extra14 = models.CharField(max_length=56, blank=True, null=True)
    extra15 = models.CharField(max_length=56, blank=True, null=True)
    extra16 = models.CharField(max_length=56, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'informacoes_extras'


class InformacoesNutricionais(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nome = models.CharField(max_length=50, blank=True, null=True)
    quantidade = models.IntegerField(blank=True, null=True)
    unidade_porcao = models.IntegerField(blank=True, null=True)
    int_medida_caseira = models.IntegerField(blank=True, null=True)
    dec_medida_caseira = models.IntegerField(blank=True, null=True)
    medida_caseira = models.IntegerField(blank=True, null=True)
    valor_calorico = models.IntegerField(blank=True, null=True)
    carboidratos = models.IntegerField(blank=True, null=True)
    carboidratos_menor_1g = models.BooleanField(blank=True, null=True)
    proteinas = models.IntegerField(blank=True, null=True)
    proteinas_menor_1g = models.BooleanField(blank=True, null=True)
    gorduras_totais = models.IntegerField(blank=True, null=True)
    gorduras_saturadas = models.IntegerField(blank=True, null=True)
    colesterol = models.IntegerField(blank=True, null=True)
    fibra = models.IntegerField(blank=True, null=True)
    fibra_menor_1g = models.BooleanField(blank=True, null=True)
    calcio = models.IntegerField(blank=True, null=True)
    ferro = models.IntegerField(blank=True, null=True)
    sodio = models.IntegerField(blank=True, null=True)
    vd_valor_calorico = models.IntegerField(blank=True, null=True)
    vd_carboidratos = models.IntegerField(blank=True, null=True)
    vd_proteinas = models.IntegerField(blank=True, null=True)
    vd_gorduras_totais = models.IntegerField(blank=True, null=True)
    vd_gorduras_saturadas = models.IntegerField(blank=True, null=True)
    vd_colesterol = models.IntegerField(blank=True, null=True)
    vd_fibra = models.IntegerField(blank=True, null=True)
    vd_calcio = models.IntegerField(blank=True, null=True)
    vd_ferro = models.IntegerField(blank=True, null=True)
    vd_sodio = models.IntegerField(blank=True, null=True)
    gorduras_trans = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    vd_gorduras_trans = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'informacoes_nutricionais'


class IpiClassificacaoEmpresa(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    indice = models.CharField(max_length=2)
    descricao = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'ipi_classificacao_empresa'


class ItemAlteracaoTributacao(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    cabecalho = models.ForeignKey(CabecalhoAlteracaoTributacao, models.DO_NOTHING, db_column='cabecalho')
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto')
    mva = models.DecimalField(max_digits=15, decimal_places=2)

    class Meta:
        managed = False
        db_table = 'item_alteracao_tributacao'


class ItemConferenciaEstoqueAlteracaoQuantidade(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    item_conferencia = models.ForeignKey('ItensConferenciaEstoque', models.DO_NOTHING, db_column='item_conferencia')
    quantidade = models.DecimalField(max_digits=20, decimal_places=3)

    class Meta:
        managed = False
        db_table = 'item_conferencia_estoque_alteracao_quantidade'


class ItemPromocao(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    promocao = models.ForeignKey('Promocao', models.DO_NOTHING, db_column='promocao')
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto')
    faixa_inicial = models.DecimalField(max_digits=65535, decimal_places=65535)
    faixa_final = models.DecimalField(max_digits=65535, decimal_places=65535)
    novo_valor = models.DecimalField(max_digits=65535, decimal_places=65535)
    desconto = models.DecimalField(max_digits=65535, decimal_places=65535)
    marcado = models.BooleanField()
    ativo = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'item_promocao'


class ItemPromocaoBrinde(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto')
    promocao = models.ForeignKey('Promocao', models.DO_NOTHING, db_column='promocao')
    marcado = models.BooleanField()
    ativo = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'item_promocao_brinde'


class ItemVendedor(models.Model):
    cod_item = models.OneToOneField('ItensOrdemServico', models.DO_NOTHING, db_column='cod_item', primary_key=True)
    cod_vendedor = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='cod_vendedor')
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'item_vendedor'
        unique_together = (('cod_item', 'cod_vendedor'),)


class ItemVendedorOs(models.Model):
    cod_item = models.ForeignKey('OsServicos', models.DO_NOTHING, db_column='cod_item', blank=True, null=True)
    cod_vendedor = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='cod_vendedor', blank=True, null=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'item_vendedor_os'


class ItensCarregamento(models.Model):
    codigo = models.AutoField(primary_key=True)
    coditem = models.ForeignKey('ItensTransferencia', models.DO_NOTHING, db_column='coditem', blank=True, null=True)
    volume = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    sequencial = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_carregamento'


class ItensClassFiscal(models.Model):
    codigo = models.AutoField(primary_key=True)
    tabela = models.ForeignKey(ClassFiscal, models.DO_NOTHING, db_column='tabela')
    uf = models.ForeignKey(Estados, models.DO_NOTHING, db_column='uf')
    aliquota = models.DecimalField(max_digits=15, decimal_places=2)
    reducao = models.DecimalField(max_digits=15, decimal_places=2)
    beneficio = models.IntegerField()
    situacao = models.IntegerField()
    tipooperacao = models.IntegerField()
    reducao2 = models.DecimalField(max_digits=15, decimal_places=2)
    cst = models.CharField(max_length=4, blank=True, null=True)
    aliquota_interna = models.DecimalField(max_digits=6, decimal_places=2)
    calcula_diferencial = models.BooleanField()
    fundo_combate_pobreza = models.DecimalField(max_digits=6, decimal_places=2)
    destino_mercadoria = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'itens_class_fiscal'


class ItensCompra(models.Model):
    codigo = models.AutoField(primary_key=True)
    pedido = models.ForeignKey('PedidoCompra', models.DO_NOTHING, db_column='pedido', blank=True, null=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535)
    quantidade_extra = models.DecimalField(max_digits=65535, decimal_places=65535)
    preco = models.DecimalField(max_digits=65535, decimal_places=65535)
    preco_extra = models.DecimalField(max_digits=65535, decimal_places=65535)
    icms = models.DecimalField(max_digits=65535, decimal_places=65535)
    reducao = models.DecimalField(max_digits=65535, decimal_places=65535)
    substituicao = models.CharField(max_length=1)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    data_lancamento = models.DateTimeField()
    cfop = models.ForeignKey(Cfop, models.DO_NOTHING, db_column='cfop', blank=True, null=True)
    descricao = models.CharField(max_length=50, blank=True, null=True)
    codigodefabrica = models.CharField(max_length=20, blank=True, null=True)
    aliquota_interna = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    uso_consumo = models.BooleanField(blank=True, null=True)
    marca = models.CharField(max_length=1, blank=True, null=True)
    observacao = models.ForeignKey('ObservacoesMovimentos', models.DO_NOTHING, db_column='observacao', blank=True, null=True)
    valoripi = models.DecimalField(max_digits=65535, decimal_places=65535)
    valorst = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_compra'


class ItensCondicional(models.Model):
    codigo = models.AutoField(primary_key=True)
    pedido = models.ForeignKey(CabecalhoCondicional, models.DO_NOTHING, db_column='pedido', blank=True, null=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    descricao = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_condicional'


class ItensConferenciaEstoque(models.Model):
    codigo = models.AutoField(primary_key=True)
    codigo_cabecalho = models.ForeignKey(CabecalhoConferenciaEstoque, models.DO_NOTHING, db_column='codigo_cabecalho', blank=True, null=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    qtd_conferida = models.DecimalField(max_digits=65535, decimal_places=65535)
    data_lancamento = models.DateField()
    hora_conferencia = models.TimeField()
    codbarra = models.CharField(max_length=14)
    utiliza_lote = models.BooleanField()
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'itens_conferencia_estoque'


class ItensDavMd5(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_md5_dav = models.IntegerField()
    item_orcamento = models.ForeignKey('ItensOrcamento', models.DO_NOTHING, db_column='item_orcamento', blank=True, null=True)
    item_venda = models.ForeignKey('ItensOrdemServico', models.DO_NOTHING, db_column='item_venda', blank=True, null=True)
    data = models.DateTimeField()
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador', blank=True, null=True)
    md5 = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_dav_md5'


class ItensDevolucao(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_devolucao = models.ForeignKey(CabecalhoDevolucao, models.DO_NOTHING, db_column='cd_devolucao', blank=True, null=True)
    cd_produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='cd_produto', blank=True, null=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    quantidade = models.DecimalField(max_digits=15, decimal_places=3, blank=True, null=True)
    produto = models.ForeignKey('ItensOrdemServico', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    baixa_estoque = models.CharField(max_length=1, blank=True, null=True)
    cfop = models.ForeignKey(Cfop, models.DO_NOTHING, db_column='cfop', blank=True, null=True)
    codbarra = models.CharField(max_length=14, blank=True, null=True)
    cd_entrega_futura = models.ForeignKey('ItensEntrega', models.DO_NOTHING, db_column='cd_entrega_futura', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_devolucao'


class ItensEnderecosEstados(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_endereco = models.ForeignKey(CabecalhoEnderecosWebservice, models.DO_NOTHING, db_column='cd_endereco')
    uf = models.ForeignKey(Estados, models.DO_NOTHING, db_column='uf')

    class Meta:
        managed = False
        db_table = 'itens_enderecos_estados'


class ItensEnderecosEstadosCte(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_endereco = models.ForeignKey(CabecalhoEnderecosWebserviceCte, models.DO_NOTHING, db_column='cd_endereco')
    uf = models.ForeignKey(Estados, models.DO_NOTHING, db_column='uf')

    class Meta:
        managed = False
        db_table = 'itens_enderecos_estados_cte'


class ItensEnderecosEstadosNfce(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_endereco = models.ForeignKey(CabecalhoEnderecosWebserviceNfce, models.DO_NOTHING, db_column='cd_endereco')
    uf = models.ForeignKey(Estados, models.DO_NOTHING, db_column='uf')

    class Meta:
        managed = False
        db_table = 'itens_enderecos_estados_nfce'


class ItensEnderecosWebservice(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_endereco = models.ForeignKey(CabecalhoEnderecosWebservice, models.DO_NOTHING, db_column='cd_endereco')
    endereco = models.CharField(max_length=500, blank=True, null=True)
    evento = models.IntegerField(blank=True, null=True)
    ambiente = models.IntegerField(blank=True, null=True)
    versaodados = models.CharField(max_length=4)

    class Meta:
        managed = False
        db_table = 'itens_enderecos_webservice'


class ItensEnderecosWebserviceCte(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_endereco = models.ForeignKey(CabecalhoEnderecosWebserviceCte, models.DO_NOTHING, db_column='cd_endereco')
    endereco = models.CharField(max_length=500, blank=True, null=True)
    evento = models.IntegerField(blank=True, null=True)
    ambiente = models.IntegerField(blank=True, null=True)
    versaodados = models.CharField(max_length=4)

    class Meta:
        managed = False
        db_table = 'itens_enderecos_webservice_cte'


class ItensEnderecosWebserviceNfce(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_endereco = models.ForeignKey(CabecalhoEnderecosWebserviceNfce, models.DO_NOTHING, db_column='cd_endereco')
    endereco = models.CharField(max_length=500, blank=True, null=True)
    evento = models.IntegerField(blank=True, null=True)
    ambiente = models.IntegerField(blank=True, null=True)
    versaodados = models.CharField(max_length=4)
    layout_qrcode = models.CharField(max_length=4)

    class Meta:
        managed = False
        db_table = 'itens_enderecos_webservice_nfce'


class ItensEntrega(models.Model):
    codigo = models.AutoField(primary_key=True)
    pedido = models.ForeignKey(CabecalhoEntrega, models.DO_NOTHING, db_column='pedido', blank=True, null=True)
    produto = models.ForeignKey('ItensOrdemServico', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=10, decimal_places=3, blank=True, null=True)
    preco_custo = models.DecimalField(max_digits=15, decimal_places=6, blank=True, null=True)
    preco_venda = models.DecimalField(max_digits=15, decimal_places=6, blank=True, null=True)
    cfop = models.ForeignKey(Cfop, models.DO_NOTHING, db_column='cfop', blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)
    data_hora = models.DateTimeField(blank=True, null=True)
    utiliza_lote = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'itens_entrega'


class ItensEntregaImpostos(models.Model):
    cd_item = models.OneToOneField(ItensEntrega, models.DO_NOTHING, db_column='cd_item', primary_key=True)
    mva = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    aliquota_intra = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    lei = models.CharField(max_length=-1, blank=True, null=True)
    cst = models.CharField(max_length=-1, blank=True, null=True)
    desconta_icms_proprio = models.BooleanField()
    icms = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    reducao = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    iss = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    ipi = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    ipi_cst = models.CharField(max_length=2, blank=True, null=True)
    ipi_enquadramento = models.CharField(max_length=3, blank=True, null=True)
    base_sub_compra = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    sub_compra = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    pis_cst = models.CharField(max_length=3, blank=True, null=True)
    cofins_cst = models.CharField(max_length=3, blank=True, null=True)
    cofins_aliq = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    pis_aliq = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_entrega_impostos'


class ItensFormula(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_item = models.ForeignKey('ItensProducao', models.DO_NOTHING, db_column='cd_item', blank=True, null=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=25, decimal_places=6, blank=True, null=True)
    descricao = models.CharField(max_length=60, blank=True, null=True)
    precocusto = models.DecimalField(max_digits=15, decimal_places=6, blank=True, null=True)
    data_lancamento = models.DateField()
    cd_item_nota = models.ForeignKey('ItensNotaCompra', models.DO_NOTHING, db_column='cd_item_nota', blank=True, null=True)
    cfop = models.ForeignKey(Cfop, models.DO_NOTHING, db_column='cfop', blank=True, null=True)
    hora_lancamento = models.TimeField()
    codbarra = models.CharField(max_length=14)
    utiliza_lote = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'itens_formula'


class ItensFormulaMedidas(models.Model):
    cd_item = models.OneToOneField(ItensFormula, models.DO_NOTHING, db_column='cd_item', primary_key=True)
    altura = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    largura = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    altura_usada = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    largura_usada = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    quantia_pecas = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_formula_medidas'


class ItensMedicamentosReferencias(models.Model):
    codigo = models.AutoField(primary_key=True)
    chave_item = models.ForeignKey('ItensNotaCompraMedicamentos', models.DO_NOTHING, db_column='chave_item', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    item_venda = models.ForeignKey('ItensOrdemServico', models.DO_NOTHING, db_column='item_venda', blank=True, null=True)
    item_venda_fora = models.ForeignKey('ItensRemessaVendaFora', models.DO_NOTHING, db_column='item_venda_fora', blank=True, null=True)
    item_retorno_fora = models.ForeignKey('ItensRetornoVendaFora', models.DO_NOTHING, db_column='item_retorno_fora', blank=True, null=True)
    item_devolucao = models.ForeignKey(ItensDevolucao, models.DO_NOTHING, db_column='item_devolucao', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_medicamentos_referencias'


class ItensModeloOrcamento(models.Model):
    codigo = models.AutoField(primary_key=True)
    modelo = models.ForeignKey(CabecalhoModeloOrcamento, models.DO_NOTHING, db_column='modelo', blank=True, null=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=8, decimal_places=2, blank=True, null=True)
    descricao = models.CharField(max_length=80, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_modelo_orcamento'


class ItensMontagem(models.Model):
    codigo = models.AutoField(primary_key=True)
    item = models.ForeignKey('ItensOrdemServico', models.DO_NOTHING, db_column='item', blank=True, null=True)
    vendedor = models.IntegerField(blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_montagem'


class ItensNotaAvulsa(models.Model):
    codigo = models.AutoField(primary_key=True)
    pedido = models.ForeignKey(CabecaNotaAvulsa, models.DO_NOTHING, db_column='pedido', blank=True, null=True)
    cd_produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='cd_produto', blank=True, null=True)
    descricao = models.CharField(max_length=100, blank=True, null=True)
    unidade = models.CharField(max_length=5, blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    total = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    cfop = models.ForeignKey(Cfop, models.DO_NOTHING, db_column='cfop', blank=True, null=True)
    destino_mercadoria = models.IntegerField(blank=True, null=True)
    itemcabecalhodespesas = models.IntegerField(blank=True, null=True)
    item_venda = models.ForeignKey('ItensOrdemServico', models.DO_NOTHING, db_column='item_venda', blank=True, null=True)
    codigo_anp = models.CharField(max_length=9, blank=True, null=True)
    iat = models.BooleanField()
    baixa_estoque = models.BooleanField()
    serial = models.ForeignKey('Seriais', models.DO_NOTHING, db_column='serial', blank=True, null=True)
    data_lancamento = models.DateField()
    hora_lancamento = models.TimeField()
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    bonificado_destaca_icms = models.BooleanField()
    acrescimo = models.DecimalField(max_digits=15, decimal_places=2)
    desconto = models.DecimalField(max_digits=15, decimal_places=2)
    utiliza_lote = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'itens_nota_avulsa'


class ItensNotaCompra(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_nota = models.ForeignKey(CabecalhoNotaCompra, models.DO_NOTHING, db_column='cd_nota', blank=True, null=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    quantidade_extra = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    preco_custo = models.DecimalField(max_digits=15, decimal_places=6, blank=True, null=True)
    custo_extra = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    desconto = models.DecimalField(max_digits=15, decimal_places=6, blank=True, null=True)
    custo_aquisicao = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    custo_minimo = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    pv_atacado = models.DecimalField(max_digits=15, decimal_places=3, blank=True, null=True)
    cfop = models.ForeignKey(Cfop, models.DO_NOTHING, db_column='cfop', blank=True, null=True)
    local = models.CharField(max_length=100, blank=True, null=True)
    p_operacional = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    conferido = models.BooleanField()
    preco_minimo = models.DecimalField(max_digits=65535, decimal_places=65535)
    ncm = models.CharField(max_length=20)
    baixa_estoque = models.CharField(max_length=1, blank=True, null=True)
    destino_mercadoria = models.IntegerField()
    bonificado = models.CharField(max_length=1)
    gta = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    peso_liquido = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    peso_bruto = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    reducao_base_st = models.DecimalField(max_digits=65535, decimal_places=65535)
    unidade = models.ForeignKey('UnidadeMedidas', models.DO_NOTHING, db_column='unidade', blank=True, null=True)
    total = models.DecimalField(max_digits=65535, decimal_places=65535)
    codigo_anp = models.CharField(max_length=9, blank=True, null=True)
    descricao = models.CharField(max_length=120)
    informacoes_adicionais_produto = models.CharField(max_length=500)
    bonificado_destaca_icms = models.BooleanField()
    utiliza_lote = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'itens_nota_compra'


class ItensNotaCompraAtualizacao(models.Model):
    codigo = models.AutoField(primary_key=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto')
    cd_item_compra = models.ForeignKey(ItensNotaCompra, models.DO_NOTHING, db_column='cd_item_compra')
    preco_contabil = models.DecimalField(max_digits=65535, decimal_places=65535)
    preco_custo = models.DecimalField(max_digits=65535, decimal_places=65535)
    preco_minimo = models.DecimalField(max_digits=65535, decimal_places=65535)
    preco_venda = models.DecimalField(max_digits=65535, decimal_places=65535)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'itens_nota_compra_atualizacao'


class ItensNotaCompraDadosImportacao(models.Model):
    cd_item = models.ForeignKey(ItensNotaCompra, models.DO_NOTHING, db_column='cd_item')
    tipo_transporte = models.IntegerField()
    valor_afrmm = models.DecimalField(max_digits=65535, decimal_places=65535)
    tipo_intermedio = models.IntegerField()
    cnpj_adquirente = models.CharField(max_length=18)
    uf_adquirente = models.CharField(max_length=2)

    class Meta:
        managed = False
        db_table = 'itens_nota_compra_dados_importacao'


class ItensNotaCompraFormacaoVenda(models.Model):
    cd_item = models.OneToOneField(ItensNotaCompra, models.DO_NOTHING, db_column='cd_item', primary_key=True)
    custo_aquisicao = models.DecimalField(max_digits=65535, decimal_places=65535)
    preco_venda = models.DecimalField(max_digits=65535, decimal_places=65535)
    markup = models.DecimalField(max_digits=65535, decimal_places=65535)
    icms = models.DecimalField(max_digits=65535, decimal_places=65535)
    pis = models.DecimalField(max_digits=65535, decimal_places=65535)
    cofins = models.DecimalField(max_digits=65535, decimal_places=65535)
    ir = models.DecimalField(max_digits=65535, decimal_places=65535)
    csll = models.DecimalField(max_digits=65535, decimal_places=65535)
    operacional = models.DecimalField(max_digits=65535, decimal_places=65535)
    outras = models.DecimalField(max_digits=65535, decimal_places=65535)
    comissao = models.DecimalField(max_digits=65535, decimal_places=65535)

    class Meta:
        managed = False
        db_table = 'itens_nota_compra_formacao_venda'


class ItensNotaCompraImportacaoXml(models.Model):
    codigo = models.AutoField(primary_key=True)
    chave_cabecalho = models.ForeignKey(CabecalhoNotaCompraImportacaoXml, models.DO_NOTHING, db_column='chave_cabecalho')
    sequencia_xml = models.IntegerField()
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto')
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535)
    preco_custo = models.DecimalField(max_digits=65535, decimal_places=65535)

    class Meta:
        managed = False
        db_table = 'itens_nota_compra_importacao_xml'


class ItensNotaCompraImpostos(models.Model):
    cd_item = models.OneToOneField(ItensNotaCompra, models.DO_NOTHING, db_column='cd_item', primary_key=True)
    cst_nota = models.CharField(max_length=4)
    cst_operacao = models.CharField(max_length=4)
    base_icms = models.DecimalField(max_digits=15, decimal_places=2)
    valor_icms = models.DecimalField(max_digits=15, decimal_places=2)
    icms = models.DecimalField(max_digits=15, decimal_places=2)
    reducao = models.DecimalField(max_digits=15, decimal_places=2)
    perc_credito = models.DecimalField(max_digits=15, decimal_places=2)
    valor_icms_credito = models.DecimalField(max_digits=15, decimal_places=2)
    base_ipi = models.DecimalField(max_digits=15, decimal_places=2)
    valor_ipi = models.DecimalField(max_digits=15, decimal_places=2)
    ipi = models.DecimalField(max_digits=15, decimal_places=2)
    ipi_enquadramento = models.CharField(max_length=3)
    ipi_cst = models.ForeignKey(CodigosFiscaisIpi, models.DO_NOTHING, db_column='ipi_cst')
    ipi_indicador = models.IntegerField()
    modalidade_icms_st = models.IntegerField()
    base_icms_st = models.DecimalField(max_digits=15, decimal_places=2)
    valor_icms_st = models.DecimalField(max_digits=15, decimal_places=2)
    mva = models.DecimalField(max_digits=15, decimal_places=2)
    icms_desconto = models.DecimalField(max_digits=15, decimal_places=2)
    reducao_mva = models.DecimalField(max_digits=15, decimal_places=2)
    base_pis = models.DecimalField(max_digits=15, decimal_places=2)
    pis_aliq = models.DecimalField(max_digits=15, decimal_places=2)
    pis_cst = models.CharField(max_length=2)
    valor_pis = models.DecimalField(max_digits=15, decimal_places=2)
    quantidade_pis = models.DecimalField(max_digits=15, decimal_places=2)
    base_cofins = models.DecimalField(max_digits=15, decimal_places=2)
    cofins_aliq = models.DecimalField(max_digits=15, decimal_places=2)
    cofins_cst = models.CharField(max_length=2)
    valor_cofins = models.DecimalField(max_digits=15, decimal_places=2)
    quantidade_cofins = models.DecimalField(max_digits=15, decimal_places=2)
    icms_importacao = models.DecimalField(max_digits=15, decimal_places=2)
    perc_importacao = models.DecimalField(max_digits=15, decimal_places=2)
    thc = models.DecimalField(max_digits=15, decimal_places=2)
    importacao = models.DecimalField(max_digits=15, decimal_places=2)
    siscomex = models.DecimalField(max_digits=15, decimal_places=2)
    aliquota_interna = models.DecimalField(max_digits=15, decimal_places=2)
    base_iss = models.DecimalField(max_digits=15, decimal_places=2)
    iss = models.DecimalField(max_digits=15, decimal_places=2)
    valor_iss = models.DecimalField(max_digits=15, decimal_places=2)
    valor_ir = models.DecimalField(max_digits=15, decimal_places=2)
    ir = models.DecimalField(max_digits=15, decimal_places=2)
    valor_csll = models.DecimalField(max_digits=15, decimal_places=2)
    csll = models.DecimalField(max_digits=15, decimal_places=2)
    valor_inss = models.DecimalField(max_digits=15, decimal_places=2)
    inss = models.DecimalField(max_digits=15, decimal_places=2)
    funrural = models.DecimalField(max_digits=15, decimal_places=2)
    reducao_base_icms_st = models.DecimalField(max_digits=15, decimal_places=2)
    valor_icms_credito_fob = models.DecimalField(max_digits=15, decimal_places=2)
    valor_credito_cofins_fob = models.DecimalField(max_digits=15, decimal_places=2)
    valor_credito_pis_fob = models.DecimalField(max_digits=15, decimal_places=2)
    adicaoimportacao = models.IntegerField()
    sequenciaimportacao = models.IntegerField()
    ipi_cst_operacao = models.ForeignKey(CodigosFiscaisIpi, models.DO_NOTHING, db_column='ipi_cst_operacao')
    seguro = models.DecimalField(max_digits=15, decimal_places=2)
    duppingbase = models.DecimalField(max_digits=15, decimal_places=2)
    duppingaliquota = models.DecimalField(max_digits=15, decimal_places=2)
    duppingvalor = models.DecimalField(max_digits=15, decimal_places=2)
    credita_icmsst = models.BooleanField()
    base_icms_st_fob = models.DecimalField(max_digits=65535, decimal_places=65535)
    valor_icms_st_fob = models.DecimalField(max_digits=65535, decimal_places=65535)
    base_icms_st_retido = models.DecimalField(max_digits=65535, decimal_places=65535)
    valor_icms_st_retido = models.DecimalField(max_digits=65535, decimal_places=65535)
    cest = models.CharField(max_length=7)
    fcp_icms_aliquota = models.DecimalField(max_digits=65535, decimal_places=65535)
    fcp_icms_valor = models.DecimalField(max_digits=65535, decimal_places=65535)
    fcp_st_icms_aliquota = models.DecimalField(max_digits=65535, decimal_places=65535)
    fcp_st_icms_valor = models.DecimalField(max_digits=65535, decimal_places=65535)
    fcp_st_aliquota_retido = models.DecimalField(max_digits=6, decimal_places=2)
    fcp_st_valor_retido = models.DecimalField(max_digits=65535, decimal_places=65535)
    percentual_st_retido = models.DecimalField(max_digits=65535, decimal_places=65535)
    cnpj_fabricante = models.CharField(max_length=18)
    beneficio = models.CharField(max_length=10)
    valor_icms_substituto = models.DecimalField(max_digits=15, decimal_places=2)
    valor_icms_original = models.DecimalField(max_digits=20, decimal_places=2)
    diferido_percentual = models.DecimalField(max_digits=20, decimal_places=2)
    diferido_valor = models.DecimalField(max_digits=20, decimal_places=2)
    desonerado_percentual = models.DecimalField(max_digits=20, decimal_places=2)
    desonerado_valor = models.DecimalField(max_digits=20, decimal_places=2)
    desonerado_desconta_total = models.BooleanField()
    desonerado_motivo = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'itens_nota_compra_impostos'


class ItensNotaCompraMedicamentos(models.Model):
    codigo = models.AutoField(primary_key=True)
    chave_item = models.ForeignKey(ItensNotaCompra, models.DO_NOTHING, db_column='chave_item', blank=True, null=True)
    lote = models.CharField(max_length=50, blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    data_fabricacao = models.DateField(blank=True, null=True)
    data_vencimento = models.DateField(blank=True, null=True)
    preco_maximo = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    codigo_anvisa = models.CharField(max_length=13)
    codigo_agregacao = models.CharField(max_length=20)

    class Meta:
        managed = False
        db_table = 'itens_nota_compra_medicamentos'


class ItensNotaCompraProporcionais(models.Model):
    cd_item = models.OneToOneField(ItensNotaCompra, models.DO_NOTHING, db_column='cd_item', primary_key=True)
    prop_frete = models.DecimalField(max_digits=65535, decimal_places=65535)
    prop_outras = models.DecimalField(max_digits=65535, decimal_places=65535)
    prop_seguro = models.DecimalField(max_digits=65535, decimal_places=65535)
    prop_frete_fob = models.DecimalField(max_digits=65535, decimal_places=65535)
    prop_fornecido = models.DecimalField(max_digits=65535, decimal_places=65535)
    prop_serv_terc = models.DecimalField(max_digits=65535, decimal_places=65535)
    prop_desp_acessorias = models.DecimalField(max_digits=65535, decimal_places=65535)

    class Meta:
        managed = False
        db_table = 'itens_nota_compra_proporcionais'


class ItensOrcamento(models.Model):
    codigo = models.AutoField(primary_key=True)
    orcamento = models.ForeignKey(CabecalhoOrcamento, models.DO_NOTHING, db_column='orcamento', blank=True, null=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    valor_unit = models.DecimalField(max_digits=65535, decimal_places=65535)
    desconto = models.DecimalField(max_digits=65535, decimal_places=65535)
    tipo_desconto = models.CharField(max_length=1)
    garantia = models.CharField(max_length=50, blank=True, null=True)
    total = models.DecimalField(max_digits=65535, decimal_places=65535)
    tipo = models.CharField(max_length=1, blank=True, null=True)
    marca = models.CharField(max_length=1, blank=True, null=True)
    descricao = models.CharField(max_length=50, blank=True, null=True)
    vendedor = models.IntegerField(blank=True, null=True)
    preco_custo = models.DecimalField(max_digits=65535, decimal_places=65535)
    custo_medio = models.DecimalField(max_digits=65535, decimal_places=65535)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535)
    separado = models.CharField(max_length=1, blank=True, null=True)
    cod_fabrica = models.CharField(max_length=20, blank=True, null=True)
    ent_futura = models.DecimalField(max_digits=65535, decimal_places=65535)
    ipi = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    cfop = models.ForeignKey(Cfop, models.DO_NOTHING, db_column='cfop', blank=True, null=True)
    consumo_proprio = models.BooleanField(blank=True, null=True)
    codbarra = models.CharField(max_length=14, blank=True, null=True)
    tabela_preco = models.ForeignKey(CabecaPreco, models.DO_NOTHING, db_column='tabela_preco', blank=True, null=True)
    industrializacao = models.BooleanField()
    acrescimo = models.DecimalField(max_digits=65535, decimal_places=65535)
    bonificado = models.BooleanField()
    preco_cmv = models.DecimalField(max_digits=65535, decimal_places=65535)
    serial = models.CharField(max_length=30)
    data_lancamento = models.DateTimeField()
    sequencia = models.IntegerField()
    md5 = models.CharField(max_length=32, blank=True, null=True)
    recarga_celular = models.BooleanField()
    bonificado_destaca_icms = models.BooleanField()
    lancou_codigo_interno = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'itens_orcamento'


class ItensOrcamentoDavD4(models.Model):
    codigo = models.OneToOneField(ItensOrcamento, models.DO_NOTHING, db_column='codigo', primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'itens_orcamento_dav_d4'


class ItensOrcamentoEntrega(models.Model):
    cd_item = models.OneToOneField(ItensOrcamento, models.DO_NOTHING, db_column='cd_item', primary_key=True)
    data_previsao = models.DateField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_orcamento_entrega'


class ItensOrcamentoExcluidosPaf(models.Model):
    cd_item = models.OneToOneField(ItensOrcamento, models.DO_NOTHING, db_column='cd_item', primary_key=True)
    data_lancamento = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'itens_orcamento_excluidos_paf'


class ItensOrcamentoImpostos(models.Model):
    cd_item = models.OneToOneField(ItensOrcamento, models.DO_NOTHING, db_column='cd_item', primary_key=True)
    mva = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    aliquota_intra = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    desconta_icms_proprio = models.BooleanField()
    lei = models.CharField(max_length=-1, blank=True, null=True)
    cst = models.CharField(max_length=-1, blank=True, null=True)
    icms = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    reducao = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    iss = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    ipi = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    pis_cst = models.CharField(max_length=3, blank=True, null=True)
    cofins_cst = models.CharField(max_length=3, blank=True, null=True)
    cofins_aliq = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    pis_aliq = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_orcamento_impostos'


class ItensOrcamentoModelo(models.Model):
    codigo = models.AutoField(primary_key=True)
    pedido = models.ForeignKey(CabecalhoOrcamentoModelo, models.DO_NOTHING, db_column='pedido', blank=True, null=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_orcamento_modelo'


class ItensOrcamentoObservacoes(models.Model):
    produto = models.OneToOneField(ItensOrcamento, models.DO_NOTHING, db_column='produto', primary_key=True)
    observacao = models.ForeignKey('ObservacoesMovimentos', models.DO_NOTHING, db_column='observacao', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_orcamento_observacoes'


class ItensOrdemServico(models.Model):
    codigo = models.AutoField(primary_key=True)
    ordem = models.ForeignKey(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', blank=True, null=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    preco_unitario = models.DecimalField(max_digits=65535, decimal_places=65535)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535)
    desconto = models.DecimalField(max_digits=65535, decimal_places=65535)
    total = models.DecimalField(max_digits=65535, decimal_places=65535)
    tipo_desconto = models.CharField(max_length=1)
    garantia = models.CharField(max_length=50)
    marca = models.CharField(max_length=1, blank=True, null=True)
    tipo = models.CharField(max_length=1)
    descricao = models.CharField(max_length=50)
    sequencia = models.IntegerField(blank=True, null=True)
    modelo = models.IntegerField(blank=True, null=True)
    preco_custo = models.DecimalField(max_digits=65535, decimal_places=65535)
    preco_contabil = models.DecimalField(max_digits=65535, decimal_places=65535)
    vendedor = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='vendedor', blank=True, null=True)
    comissao = models.DecimalField(max_digits=10, decimal_places=3, blank=True, null=True)
    e_futura = models.DecimalField(max_digits=65535, decimal_places=65535)
    tabela_preco = models.IntegerField(blank=True, null=True)
    preco_partida = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    cfop = models.ForeignKey(Cfop, models.DO_NOTHING, db_column='cfop', blank=True, null=True)
    operador = models.IntegerField()
    data_lancamento = models.DateField()
    hora_lancamento = models.TimeField()
    consumo_proprio = models.BooleanField(blank=True, null=True)
    acrescimo = models.DecimalField(max_digits=65535, decimal_places=65535)
    baixa_estoque = models.CharField(max_length=1, blank=True, null=True)
    iat = models.BooleanField()
    codbarra = models.CharField(max_length=14, blank=True, null=True)
    industrializacao = models.BooleanField()
    md5 = models.CharField(max_length=32)
    bonificado = models.BooleanField()
    recarga_celular = models.BooleanField()
    temestoque = models.BooleanField()
    md5dav = models.CharField(max_length=32, blank=True, null=True)
    bonificado_destaca_icms = models.BooleanField()
    lancou_codigo_interno = models.BooleanField()
    utiliza_lote = models.BooleanField()
    promocao = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'itens_ordem_servico'


class ItensOrdemServicoCidadeServico(models.Model):
    item = models.ForeignKey(ItensOrdemServico, models.DO_NOTHING, db_column='item', blank=True, null=True)
    cidade = models.ForeignKey(Cidades, models.DO_NOTHING, db_column='cidade')
    data = models.DateTimeField()
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')

    class Meta:
        managed = False
        db_table = 'itens_ordem_servico_cidade_servico'


class ItensOrdemServicoComposicaoValor(models.Model):
    codigo = models.AutoField(primary_key=True)
    ordem = models.ForeignKey(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', blank=True, null=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto')
    precovenda = models.DecimalField(max_digits=65535, decimal_places=65535)
    diferenca_composicao = models.DecimalField(max_digits=65535, decimal_places=65535)
    pid = models.BigIntegerField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    datahora = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'itens_ordem_servico_composicao_valor'


class ItensOrdemServicoCustos(models.Model):
    codigo = models.AutoField(primary_key=True)
    custo_cmv = models.DecimalField(max_digits=65535, decimal_places=65535)
    custo_normal = models.DecimalField(max_digits=65535, decimal_places=65535)
    item_ordem_servico = models.ForeignKey(ItensOrdemServico, models.DO_NOTHING, db_column='item_ordem_servico', blank=True, null=True)
    item_orcamento = models.ForeignKey(ItensOrcamento, models.DO_NOTHING, db_column='item_orcamento', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_ordem_servico_custos'


class ItensOrdemServicoDescricao(models.Model):
    cd_item = models.OneToOneField(ItensOrdemServico, models.DO_NOTHING, db_column='cd_item', primary_key=True)
    descricao = models.CharField(max_length=10000000, blank=True, null=True)
    data = models.DateTimeField()
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')

    class Meta:
        managed = False
        db_table = 'itens_ordem_servico_descricao'


class ItensOrdemServicoEntrega(models.Model):
    cd_item = models.OneToOneField(ItensOrdemServico, models.DO_NOTHING, db_column='cd_item', primary_key=True)
    data_previsao = models.DateField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_ordem_servico_entrega'


class ItensOrdemServicoExcluidos(models.Model):
    codigo = models.AutoField(primary_key=True)
    ordem = models.IntegerField(blank=True, null=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    descricao = models.CharField(max_length=-1, blank=True, null=True)
    unitario = models.DecimalField(max_digits=65535, decimal_places=65535)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535)
    total = models.DecimalField(max_digits=65535, decimal_places=65535)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    data = models.DateTimeField()
    sequencia = models.IntegerField(blank=True, null=True)
    usuarioautorizou = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuarioautorizou')
    datahoraautorizacao = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'itens_ordem_servico_excluidos'


class ItensOrdemServicoExcluidosPaf(models.Model):
    cd_item = models.OneToOneField(ItensOrdemServico, models.DO_NOTHING, db_column='cd_item', primary_key=True)

    class Meta:
        managed = False
        db_table = 'itens_ordem_servico_excluidos_paf'


class ItensOrdemServicoExportacao(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_item = models.ForeignKey(ItensOrdemServico, models.DO_NOTHING, db_column='cd_item')
    numero_drawback = models.BigIntegerField()
    numero_registro_exportacao = models.BigIntegerField()
    chave_nfe_recebida = models.CharField(max_length=44)
    quantidade_exportada = models.DecimalField(max_digits=11, decimal_places=4)

    class Meta:
        managed = False
        db_table = 'itens_ordem_servico_exportacao'


class ItensOrdemServicoFaixa(models.Model):
    cd_item = models.OneToOneField(ItensOrdemServico, models.DO_NOTHING, db_column='cd_item', primary_key=True)
    faixa = models.IntegerField(blank=True, null=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    data = models.DateTimeField()
    validade_inicial = models.DateTimeField()
    validade_final = models.DateTimeField()
    faixa_final = models.DecimalField(max_digits=65535, decimal_places=65535)
    novo_valor = models.DecimalField(max_digits=65535, decimal_places=65535)
    faixa_inicial = models.DecimalField(max_digits=65535, decimal_places=65535)

    class Meta:
        managed = False
        db_table = 'itens_ordem_servico_faixa'


class ItensOrdemServicoFardo(models.Model):
    cd_item = models.OneToOneField(ItensOrdemServico, models.DO_NOTHING, db_column='cd_item', primary_key=True)
    tipo_fardo = models.IntegerField()
    peso = models.DecimalField(max_digits=65535, decimal_places=65535)
    peso_total = models.DecimalField(max_digits=65535, decimal_places=65535)
    fardo = models.DecimalField(max_digits=65535, decimal_places=65535)

    class Meta:
        managed = False
        db_table = 'itens_ordem_servico_fardo'


class ItensOrdemServicoIbpt(models.Model):
    cd_item = models.OneToOneField(ItensOrdemServico, models.DO_NOTHING, db_column='cd_item', primary_key=True)
    percentual = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    valor = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    percentual_estadual = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    valor_estadual = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    percentual_municipal = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    valor_municipal = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    chave = models.CharField(max_length=-1, blank=True, null=True)
    fonte = models.CharField(max_length=-1, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_ordem_servico_ibpt'


class ItensOrdemServicoImpostosExcluidos(models.Model):
    cd_item = models.OneToOneField(ItensOrdemServicoExcluidos, models.DO_NOTHING, db_column='cd_item', primary_key=True)
    mva = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    aliquota_intra = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    cst = models.CharField(max_length=-1, blank=True, null=True)
    desconta_icms_proprio = models.BooleanField()
    icms = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    reducao = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    iss = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    ipi = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    ipi_cst = models.CharField(max_length=2, blank=True, null=True)
    ipi_enquadramento = models.CharField(max_length=3, blank=True, null=True)
    base_sub_compra = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    sub_compra = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    pis_cst = models.CharField(max_length=3, blank=True, null=True)
    cofins_cst = models.CharField(max_length=3, blank=True, null=True)
    cofins_aliq = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    pis_aliq = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    md5 = models.CharField(max_length=-1)

    class Meta:
        managed = False
        db_table = 'itens_ordem_servico_impostos_excluidos'


class ItensOrdemServicoImpostosMd5(models.Model):
    cd_item = models.OneToOneField(ItensOrdemServico, models.DO_NOTHING, db_column='cd_item', primary_key=True)
    md5 = models.CharField(max_length=-1)

    class Meta:
        managed = False
        db_table = 'itens_ordem_servico_impostos_md5'


class ItensOrdemServicoKit(models.Model):
    codigo = models.AutoField(primary_key=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    datahora = models.DateTimeField()
    item = models.ForeignKey(ItensOrdemServico, models.DO_NOTHING, db_column='item')
    composicao = models.ForeignKey(ItensOrdemServicoComposicaoValor, models.DO_NOTHING, db_column='composicao')

    class Meta:
        managed = False
        db_table = 'itens_ordem_servico_kit'


class ItensOrdemServicoMd5J2(models.Model):
    cd_item = models.ForeignKey(ItensOrdemServico, models.DO_NOTHING, db_column='cd_item', blank=True, null=True)
    md5 = models.CharField(max_length=32, blank=True, null=True)
    cd_item_producao_entrada = models.ForeignKey('ItensProducao', models.DO_NOTHING, db_column='cd_item_producao_entrada', blank=True, null=True)
    cd_item_producao = models.ForeignKey(ItensFormula, models.DO_NOTHING, db_column='cd_item_producao', blank=True, null=True)
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'itens_ordem_servico_md5_j2'


class ItensOrdemServicoMedidas(models.Model):
    cd_item = models.OneToOneField(ItensOrdemServico, models.DO_NOTHING, db_column='cd_item', primary_key=True)
    altura = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    largura = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    altura_usada = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    largura_usada = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    quantia_pecas = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    altura_cobrada = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    comprimento_cobrado = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_ordem_servico_medidas'


class ItensOrdemServicoOs(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_item = models.ForeignKey(ItensOrdemServico, models.DO_NOTHING, db_column='cd_item', blank=True, null=True)
    ordem = models.ForeignKey(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_ordem_servico_os'


class ItensOrdemServicoPedidoComprador(models.Model):
    codigo = models.OneToOneField(ItensOrdemServico, models.DO_NOTHING, db_column='codigo', primary_key=True)
    pedido = models.CharField(max_length=15)
    item_ped = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'itens_ordem_servico_pedido_comprador'


class ItensOrdemServicoPontos(models.Model):
    cd_item = models.OneToOneField(ItensOrdemServico, models.DO_NOTHING, db_column='cd_item', primary_key=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535)
    indexador = models.DecimalField(max_digits=65535, decimal_places=65535)
    pontos = models.DecimalField(max_digits=65535, decimal_places=65535)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'itens_ordem_servico_pontos'


class ItensOrdemServicoProducao(models.Model):
    codigo = models.AutoField(primary_key=True)
    item = models.ForeignKey(ItensOrdemServico, models.DO_NOTHING, db_column='item', blank=True, null=True)
    producao = models.ForeignKey('OrdemProducao', models.DO_NOTHING, db_column='producao', blank=True, null=True)
    data_hora = models.DateTimeField(blank=True, null=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_ordem_servico_producao'


class ItensOrdemServicoRemessaVendaFora(models.Model):
    item_venda = models.ForeignKey(ItensOrdemServico, models.DO_NOTHING, db_column='item_venda', blank=True, null=True)
    item_remessa = models.ForeignKey('ItensRemessaVendaFora', models.DO_NOTHING, db_column='item_remessa', blank=True, null=True)
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'itens_ordem_servico_remessa_venda_fora'


class ItensOrdemServicoRestaurante(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_item = models.ForeignKey(ItensOrdemServico, models.DO_NOTHING, db_column='cd_item', blank=True, null=True)
    ordem = models.ForeignKey(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_ordem_servico_restaurante'


class ItensOrdemServicoSaldoEtiquetas(models.Model):
    codigo = models.OneToOneField(ItensOrdemServico, models.DO_NOTHING, db_column='codigo', primary_key=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535)

    class Meta:
        managed = False
        db_table = 'itens_ordem_servico_saldo_etiquetas'


class ItensPedidoCompra(models.Model):
    codigo = models.AutoField(primary_key=True)
    pedido = models.ForeignKey(CabecalhoPedidoCompra, models.DO_NOTHING, db_column='pedido', blank=True, null=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    total = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    valor_unit = models.DecimalField(max_digits=12, decimal_places=2, blank=True, null=True)
    descricao = models.CharField(max_length=80, blank=True, null=True)
    marca = models.CharField(max_length=1, blank=True, null=True)
    quantidade = models.DecimalField(max_digits=12, decimal_places=3, blank=True, null=True)
    extra = models.DecimalField(max_digits=10, decimal_places=3, blank=True, null=True)
    codigo_fabrica = models.CharField(max_length=20, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_pedido_compra'


class ItensPedidoCompraSaldo(models.Model):
    codigo = models.OneToOneField(ItensCompra, models.DO_NOTHING, db_column='codigo', primary_key=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535)
    saldo = models.DecimalField(max_digits=65535, decimal_places=65535)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    data = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_pedido_compra_saldo'


class ItensPerda(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_item = models.ForeignKey('ItensProducao', models.DO_NOTHING, db_column='cd_item', blank=True, null=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=25, decimal_places=6, blank=True, null=True)
    descricao = models.CharField(max_length=60, blank=True, null=True)
    precocusto = models.DecimalField(max_digits=15, decimal_places=6, blank=True, null=True)
    data_lancamento = models.DateField()
    cfop = models.ForeignKey(Cfop, models.DO_NOTHING, db_column='cfop', blank=True, null=True)
    codbarra = models.CharField(max_length=14)
    utiliza_lote = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'itens_perda'


class ItensPreco(models.Model):
    codigo = models.AutoField(primary_key=True)
    tabela = models.ForeignKey(CabecaPreco, models.DO_NOTHING, db_column='tabela')
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto')
    valor = models.DecimalField(max_digits=65535, decimal_places=65535)
    tipo = models.CharField(max_length=1)
    comissao = models.DecimalField(max_digits=10, decimal_places=3)
    comissao_prazo = models.DecimalField(max_digits=10, decimal_places=3)
    data_alteracao = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'itens_preco'


class ItensPrecoDesativaPalm(models.Model):
    codigo = models.AutoField(primary_key=True)
    data = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    produto = models.IntegerField(blank=True, null=True)
    tabela = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_preco_desativa_palm'


class ItensProducao(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_ordem = models.ForeignKey('OrdemProducao', models.DO_NOTHING, db_column='cd_ordem', blank=True, null=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    descricao = models.CharField(max_length=60, blank=True, null=True)
    precocusto = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    fator = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    lote = models.CharField(max_length=10, blank=True, null=True)
    quant_formula = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    quantidade_adicional = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    utiliza_lote = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'itens_producao'


class ItensProducaoReferenciaVenda(models.Model):
    codigo = models.AutoField(primary_key=True)
    item_venda = models.ForeignKey(ItensOrdemServico, models.DO_NOTHING, db_column='item_venda')
    item_producao = models.ForeignKey(ItensProducao, models.DO_NOTHING, db_column='item_producao')
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    datahora = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'itens_producao_referencia_venda'


class ItensProporcionais(models.Model):
    frete = models.DecimalField(max_digits=15, decimal_places=2)
    seguro = models.DecimalField(max_digits=15, decimal_places=2)
    outras = models.DecimalField(max_digits=15, decimal_places=2)
    acrescimo = models.DecimalField(max_digits=15, decimal_places=2)
    desconto = models.DecimalField(max_digits=15, decimal_places=2)
    cd_imposto = models.OneToOneField(Impostos, models.DO_NOTHING, db_column='cd_imposto', primary_key=True)
    desconto_pontos = models.DecimalField(max_digits=15, decimal_places=2)

    class Meta:
        managed = False
        db_table = 'itens_proporcionais'


class ItensRemessa(models.Model):
    codigo = models.AutoField(primary_key=True)
    numero = models.ForeignKey('Remessa', models.DO_NOTHING, db_column='numero', blank=True, null=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=20, decimal_places=3, blank=True, null=True)
    icms = models.DecimalField(max_digits=6, decimal_places=2, blank=True, null=True)
    descricao = models.CharField(max_length=50, blank=True, null=True)
    preco_custo = models.DecimalField(max_digits=20, decimal_places=6, blank=True, null=True)
    reducao = models.DecimalField(max_digits=6, decimal_places=2, blank=True, null=True)
    baixa_estoque = models.CharField(max_length=1, blank=True, null=True)
    cfop = models.ForeignKey(Cfop, models.DO_NOTHING, db_column='cfop', blank=True, null=True)
    item_compra = models.ForeignKey(ItensNotaCompra, models.DO_NOTHING, db_column='item_compra', blank=True, null=True)
    data_hora_lancamento = models.DateTimeField(blank=True, null=True)
    codbarra = models.CharField(max_length=14)
    utiliza_lote = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'itens_remessa'


class ItensRemessaEntrada(models.Model):
    codigo = models.AutoField(primary_key=True)
    numero = models.ForeignKey('RemessaEntrada', models.DO_NOTHING, db_column='numero', blank=True, null=True)
    numero_saida = models.ForeignKey('Remessa', models.DO_NOTHING, db_column='numero_saida', blank=True, null=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    descricao = models.CharField(max_length=50, blank=True, null=True)
    produto_saida = models.IntegerField(blank=True, null=True)
    cod_item_remessa = models.ForeignKey(ItensRemessa, models.DO_NOTHING, db_column='cod_item_remessa', blank=True, null=True)
    data_hora_lancamento = models.DateTimeField(blank=True, null=True)
    utiliza_lote = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'itens_remessa_entrada'


class ItensRemessaImpostos(models.Model):
    cd_item = models.OneToOneField(ItensRemessa, models.DO_NOTHING, db_column='cd_item', primary_key=True)
    mva = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    aliquota_intra = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    cst = models.CharField(max_length=-1, blank=True, null=True)
    icms = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    reducao = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    ipi = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    ipi_cst = models.CharField(max_length=2, blank=True, null=True)
    ipi_enquadramento = models.CharField(max_length=3, blank=True, null=True)
    pis_cst = models.CharField(max_length=3, blank=True, null=True)
    cofins_cst = models.CharField(max_length=3, blank=True, null=True)
    cofins_aliq = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    pis_aliq = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_remessa_impostos'


class ItensRemessaVendaFora(models.Model):
    codigo = models.AutoField(primary_key=True)
    pedido = models.ForeignKey(CabecalhoRemessaVendaFora, models.DO_NOTHING, db_column='pedido', blank=True, null=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535)
    preco_unitario = models.DecimalField(max_digits=65535, decimal_places=65535)
    tipo_desconto = models.CharField(max_length=1, blank=True, null=True)
    desconto_item = models.DecimalField(max_digits=65535, decimal_places=65535)
    preco_total = models.DecimalField(max_digits=65535, decimal_places=65535)
    data_lancamento = models.DateField()
    hora_lancamento = models.TimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    baixa = models.BooleanField()
    cfop = models.ForeignKey(Cfop, models.DO_NOTHING, db_column='cfop', blank=True, null=True)
    baixa_estoque = models.CharField(max_length=1)

    class Meta:
        managed = False
        db_table = 'itens_remessa_venda_fora'


class ItensRemessaVendaForaImpostos(models.Model):
    produto = models.OneToOneField(ItensRemessaVendaFora, models.DO_NOTHING, db_column='produto', primary_key=True)
    icms = models.DecimalField(max_digits=65535, decimal_places=65535)
    iss = models.DecimalField(max_digits=65535, decimal_places=65535)
    reducao = models.DecimalField(max_digits=65535, decimal_places=65535)
    mva = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    aliquota_intra = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    desconta_icms_proprio = models.BooleanField(blank=True, null=True)
    lei = models.CharField(max_length=-1, blank=True, null=True)
    cst = models.CharField(max_length=-1, blank=True, null=True)
    ipi = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    ipi_cst = models.CharField(max_length=-1, blank=True, null=True)
    ipi_enquadramento = models.CharField(max_length=3, blank=True, null=True)
    pis_cst = models.CharField(max_length=3, blank=True, null=True)
    cofins_cst = models.CharField(max_length=3, blank=True, null=True)
    cofins_aliq = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    pis_aliq = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_remessa_venda_fora_impostos'


class ItensRemessaVendaForaSaldo(models.Model):
    codigo = models.OneToOneField(ItensRemessaVendaFora, models.DO_NOTHING, db_column='codigo', primary_key=True)
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente')
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto')
    vendedor = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='vendedor')
    cfop = models.ForeignKey(Cfop, models.DO_NOTHING, db_column='cfop', blank=True, null=True)
    nota = models.IntegerField(blank=True, null=True)
    data = models.DateTimeField()
    descricao = models.CharField(max_length=-1)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535)
    saldo = models.DecimalField(max_digits=65535, decimal_places=65535)

    class Meta:
        managed = False
        db_table = 'itens_remessa_venda_fora_saldo'


class ItensReservaSolicitados(models.Model):
    cd_item = models.OneToOneField(ItensOrcamento, models.DO_NOTHING, db_column='cd_item', primary_key=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535)
    data_hora = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'itens_reserva_solicitados'


class ItensReservasAlterados(models.Model):
    cd_item = models.OneToOneField(ItensOrcamento, models.DO_NOTHING, db_column='cd_item', primary_key=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535)
    data_hora = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'itens_reservas_alterados'


class ItensReservasExcluidos(models.Model):
    codigo = models.AutoField(primary_key=True)
    orcamento = models.ForeignKey(CabecalhoOrcamento, models.DO_NOTHING, db_column='orcamento', blank=True, null=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535)
    total = models.DecimalField(max_digits=65535, decimal_places=65535)
    data_hora = models.DateTimeField()
    bonificado = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'itens_reservas_excluidos'


class ItensRetornoVendaFora(models.Model):
    codigo = models.AutoField(primary_key=True)
    pedido = models.ForeignKey(CabecalhoRetornoVendaFora, models.DO_NOTHING, db_column='pedido', blank=True, null=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535)
    preco_unitario = models.DecimalField(max_digits=65535, decimal_places=65535)
    tipo_desconto = models.CharField(max_length=1)
    desconto_item = models.DecimalField(max_digits=65535, decimal_places=65535)
    preco_total = models.DecimalField(max_digits=65535, decimal_places=65535)
    data_lancamento = models.DateField()
    hora_lancamento = models.TimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    cd_item_remessa = models.ForeignKey(ItensRemessaVendaFora, models.DO_NOTHING, db_column='cd_item_remessa', blank=True, null=True)
    baixa = models.BooleanField()
    cfop = models.ForeignKey(Cfop, models.DO_NOTHING, db_column='cfop', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_retorno_venda_fora'


class ItensRetornoVendaForaImpostos(models.Model):
    produto = models.OneToOneField(ItensRetornoVendaFora, models.DO_NOTHING, db_column='produto', primary_key=True)
    icms = models.DecimalField(max_digits=65535, decimal_places=65535)
    iss = models.DecimalField(max_digits=65535, decimal_places=65535)
    reducao = models.DecimalField(max_digits=65535, decimal_places=65535)

    class Meta:
        managed = False
        db_table = 'itens_retorno_venda_fora_impostos'


class ItensRomaneio(models.Model):
    codigo = models.AutoField(primary_key=True)
    romaneio = models.ForeignKey(CabecalhoRomaneio, models.DO_NOTHING, db_column='romaneio')
    ordem = models.IntegerField()
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador', blank=True, null=True)
    data_hora = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'itens_romaneio'


class ItensRomaneioEntregue(models.Model):
    item_romaneio = models.OneToOneField(ItensRomaneio, models.DO_NOTHING, db_column='item_romaneio', primary_key=True)
    dataentrega = models.DateField()
    horaentrega = models.TimeField()
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    data_hora = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'itens_romaneio_entregue'


class ItensRomaneioMarcados(models.Model):
    item_romaneio = models.OneToOneField(ItensRomaneio, models.DO_NOTHING, db_column='item_romaneio', primary_key=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador', blank=True, null=True)
    data_hora = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'itens_romaneio_marcados'


class ItensRomaneioObservacoes(models.Model):
    item_romaneio = models.OneToOneField(ItensRomaneio, models.DO_NOTHING, db_column='item_romaneio', primary_key=True)
    observacao = models.TextField(blank=True, null=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador', blank=True, null=True)
    data_hora = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'itens_romaneio_observacoes'


class ItensRomaneioTransferido(models.Model):
    codigo = models.AutoField(primary_key=True)
    item_romaneio_origem = models.ForeignKey(ItensRomaneio, models.DO_NOTHING, db_column='item_romaneio_origem')
    item_romaneio_destino = models.ForeignKey(ItensRomaneio, models.DO_NOTHING, db_column='item_romaneio_destino')
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    data_hora = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'itens_romaneio_transferido'


class ItensSeparacaoEtiquetas(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_separacao = models.ForeignKey(CabecalhoSeparacaoEtiquetas, models.DO_NOTHING, db_column='cd_separacao')
    cd_item = models.ForeignKey(ItensOrdemServico, models.DO_NOTHING, db_column='cd_item')
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535)
    volume = models.IntegerField()
    data_operacao = models.DateTimeField()
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')

    class Meta:
        managed = False
        db_table = 'itens_separacao_etiquetas'


class ItensTabelasPromocionais(models.Model):
    codigo = models.AutoField(primary_key=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    tabela = models.ForeignKey(CabecalhoTabelasPromocionais, models.DO_NOTHING, db_column='tabela', blank=True, null=True)
    preco_original = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    preco_novo = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    comissao = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    comissao_prazo = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    preco_minimo = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    preco_minimo_novo = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_tabelas_promocionais'


class ItensTransferencia(models.Model):
    ordem = models.AutoField(primary_key=True)
    codpedido = models.ForeignKey('Transferencia', models.DO_NOTHING, db_column='codpedido', blank=True, null=True)
    cod_produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='cod_produto', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    custo_produto = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    baixa_estoque = models.CharField(max_length=1, blank=True, null=True)
    cfop = models.ForeignKey(Cfop, models.DO_NOTHING, db_column='cfop', blank=True, null=True)
    datahora = models.DateTimeField()
    codbarra = models.CharField(max_length=14)
    utiliza_lote = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'itens_transferencia'


class ItensTransferenciaEntrada(models.Model):
    ordem = models.AutoField(primary_key=True)
    codpedido = models.ForeignKey('TransferenciaEntrada', models.DO_NOTHING, db_column='codpedido', blank=True, null=True)
    codfilial = models.IntegerField(blank=True, null=True)
    cod_produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='cod_produto', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=10, decimal_places=3, blank=True, null=True)
    quantidade_dg = models.DecimalField(max_digits=10, decimal_places=3, blank=True, null=True)
    custo_produto = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    utiliza_lote = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'itens_transferencia_entrada'


class ItensTransferenciaImpostos(models.Model):
    cd_item = models.OneToOneField(ItensTransferencia, models.DO_NOTHING, db_column='cd_item', primary_key=True)
    mva = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    aliquota_intra = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    cst = models.CharField(max_length=4, blank=True, null=True)
    desconta_icms_proprio = models.BooleanField()
    icms = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    reducao = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    ipi = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    lei = models.CharField(max_length=-1, blank=True, null=True)
    ipi_cst = models.CharField(max_length=2, blank=True, null=True)
    ipi_enquadramento = models.CharField(max_length=3, blank=True, null=True)
    pis_cst = models.CharField(max_length=2, blank=True, null=True)
    cofins_cst = models.CharField(max_length=2, blank=True, null=True)
    cofins_aliq = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    pis_aliq = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'itens_transferencia_impostos'


class LancamentoFolha(models.Model):
    codigo = models.AutoField(primary_key=True)
    conta = models.ForeignKey(ContasFolha, models.DO_NOTHING, db_column='conta', blank=True, null=True)
    funcionario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='funcionario', blank=True, null=True)
    complemento = models.CharField(max_length=100, blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    data_lanc = models.DateTimeField(blank=True, null=True)
    usuario = models.IntegerField(blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535)
    quite = models.CharField(max_length=1, blank=True, null=True)
    operador_quitou = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador_quitou', blank=True, null=True)
    data_quitou = models.DateField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'lancamento_folha'


class Lctbancarios(models.Model):
    codigo = models.AutoField(primary_key=True)
    ctabancaria = models.ForeignKey(Ctabancarias, models.DO_NOTHING, db_column='ctabancaria', blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    numcheque = models.CharField(max_length=16, blank=True, null=True)
    codhis = models.ForeignKey(Historicos, models.DO_NOTHING, db_column='codhis', blank=True, null=True)
    valor = models.DecimalField(max_digits=15, decimal_places=2)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador', blank=True, null=True)
    datalancamento = models.DateField(blank=True, null=True)
    horalancamento = models.TimeField(blank=True, null=True)
    origem = models.IntegerField(blank=True, null=True)
    item_origem = models.IntegerField(blank=True, null=True)
    complemento = models.CharField(max_length=200, blank=True, null=True)
    marca = models.CharField(max_length=1, blank=True, null=True)
    parcela = models.ForeignKey('Parcelas', models.DO_NOTHING, db_column='parcela', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'lctbancarios'


class LctbancariosConciliacao(models.Model):
    codigo = models.AutoField()
    conta = models.ForeignKey(Ctabancarias, models.DO_NOTHING, db_column='conta', blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    idlancamento = models.CharField(max_length=-1, blank=True, null=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    descricao = models.CharField(max_length=-1, blank=True, null=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')

    class Meta:
        managed = False
        db_table = 'lctbancarios_conciliacao'


class LeituraxImpressa(models.Model):
    numero_serie = models.CharField(max_length=-1)
    data = models.DateField()
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'leiturax_impressa'


class LogAlteracaoCompra(models.Model):
    codigo = models.AutoField(primary_key=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    data_hora = models.DateTimeField()
    nota = models.ForeignKey(CabecalhoNotaCompra, models.DO_NOTHING, db_column='nota', blank=True, null=True)
    item = models.ForeignKey(ItensNotaCompra, models.DO_NOTHING, db_column='item', blank=True, null=True)
    campo = models.CharField(max_length=-1)
    valor_antigo = models.CharField(max_length=-1, blank=True, null=True)
    valor_novo = models.CharField(max_length=-1, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'log_alteracao_compra'


class LogAlteracoesBoletos(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_convenio = models.ForeignKey(ConvenioBancos, models.DO_NOTHING, db_column='cd_convenio')
    aceite = models.IntegerField(blank=True, null=True)
    carteira = models.IntegerField(blank=True, null=True)
    cd_movimento = models.IntegerField(blank=True, null=True)
    especie = models.IntegerField(blank=True, null=True)
    emite_boleto = models.IntegerField(blank=True, null=True)
    distr_boleto = models.IntegerField(blank=True, null=True)
    protestar = models.IntegerField(blank=True, null=True)
    dia_protesto = models.IntegerField(blank=True, null=True)
    mensagem1 = models.CharField(max_length=40, blank=True, null=True)
    mensagem2 = models.CharField(max_length=40, blank=True, null=True)
    mensagem3 = models.CharField(max_length=40, blank=True, null=True)
    mensagem4 = models.CharField(max_length=40, blank=True, null=True)
    mensagem5 = models.CharField(max_length=40, blank=True, null=True)
    remessa = models.IntegerField(blank=True, null=True)
    tipo_documento = models.IntegerField(blank=True, null=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'log_alteracoes_boletos'


class LogAlteracoesPdv(models.Model):
    serie_ecf = models.CharField(max_length=-1, blank=True, null=True)
    terminal = models.IntegerField(blank=True, null=True)
    motivo = models.CharField(max_length=-1, blank=True, null=True)
    gt_anterior = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    gt_atual = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'log_alteracoes_pdv'


class LogCabecalhoMetas(models.Model):
    codigo = models.AutoField()
    codigo_competencia = models.IntegerField(blank=True, null=True)
    competencia_inicial = models.DateField(blank=True, null=True)
    competencia_final = models.DateField(blank=True, null=True)
    descricao = models.CharField(max_length=50, blank=True, null=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    operacao = models.CharField(max_length=1, blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    data_hora = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'log_cabecalho_metas'


class LogCancelamentoNfe(models.Model):
    chave_nota = models.ForeignKey('NotasEletronicas', models.DO_NOTHING, db_column='chave_nota', blank=True, null=True)
    data = models.DateField()
    hora = models.TimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'log_cancelamento_nfe'


class LogCfop(models.Model):
    codigo = models.AutoField(primary_key=True)
    cfop = models.CharField(max_length=20, blank=True, null=True)
    descricao = models.CharField(max_length=50, blank=True, null=True)
    obs_nota = models.CharField(max_length=100, blank=True, null=True)
    obs_geral = models.CharField(max_length=100, blank=True, null=True)
    nota_tributada = models.CharField(max_length=1, blank=True, null=True)
    movimenta_estoque = models.CharField(max_length=1, blank=True, null=True)
    acumulador1 = models.IntegerField(blank=True, null=True)
    acumulador2 = models.IntegerField(blank=True, null=True)
    bonificado = models.CharField(max_length=1, blank=True, null=True)
    atualizacao = models.BooleanField(blank=True, null=True)
    ativo = models.BooleanField(blank=True, null=True)
    data = models.DateField()
    hora = models.TimeField()
    vendedor = models.IntegerField()
    evento = models.CharField(max_length=1)
    bonificado_destaca_icms = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'log_cfop'


class LogDesativacaoPdv(models.Model):
    codigo = models.AutoField(primary_key=True)
    terminal = models.ForeignKey('Terminais', models.DO_NOTHING, db_column='terminal', blank=True, null=True)
    datahoradesativacao = models.DateTimeField()
    usuariodesativacao = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuariodesativacao', blank=True, null=True)
    datahoraativacao = models.DateTimeField(blank=True, null=True)
    usuarioativacao = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuarioativacao', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'log_desativacao_pdv'


class LogDuplicatasEmitidas(models.Model):
    codigo = models.AutoField(primary_key=True)
    cod_parcela = models.IntegerField(blank=True, null=True)
    valor = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)
    data = models.DateField()
    hora = models.TimeField()
    datahoraexclusao = models.DateTimeField()
    usuarioexclusao = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuarioexclusao')

    class Meta:
        managed = False
        db_table = 'log_duplicatas_emitidas'


class LogEnvioBi(models.Model):
    codigo = models.AutoField()
    tabela = models.CharField(max_length=30)
    data = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'log_envio_bi'


class LogImagemProduto(models.Model):
    produto = models.OneToOneField('Produtos', models.DO_NOTHING, db_column='produto', primary_key=True)
    data = models.DateTimeField(blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'log_imagem_produto'


class LogInformacoesSpc(models.Model):
    codigo = models.AutoField(primary_key=True)
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente', blank=True, null=True)
    data = models.DateTimeField()
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    operador_reabilitacao = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador_reabilitacao', blank=True, null=True)
    data_realibitacao = models.DateField(blank=True, null=True)
    negativado = models.DateField(blank=True, null=True)
    ativado = models.DateField(blank=True, null=True)
    sistema_protecao = models.ForeignKey(Fornecedores, models.DO_NOTHING, db_column='sistema_protecao', blank=True, null=True)
    protocolo_desativacao = models.CharField(max_length=-1)
    protocolo_ativacao = models.CharField(max_length=-1)

    class Meta:
        managed = False
        db_table = 'log_informacoes_spc'


class LogNotas(models.Model):
    data = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)
    erro = models.CharField(max_length=-1, blank=True, null=True)
    numero_nota = models.IntegerField(blank=True, null=True)
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'log_notas'


class LogOcorrencias(models.Model):
    codigo = models.AutoField(primary_key=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    data_operacao = models.DateTimeField()
    tabela = models.CharField(max_length=-1)
    ocorrencia = models.CharField(max_length=-1)

    class Meta:
        managed = False
        db_table = 'log_ocorrencias'


class LogOcorrenciasDetalhes(models.Model):
    codigo = models.ForeignKey(LogOcorrencias, models.DO_NOTHING, db_column='codigo')
    numero_campo = models.IntegerField()
    nome_campo = models.CharField(max_length=-1)
    valor = models.CharField(max_length=-1, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'log_ocorrencias_detalhes'


class LogPdv(models.Model):
    codigo = models.AutoField(primary_key=True)
    erro = models.CharField(max_length=65535, blank=True, null=True)
    data = models.DateField()
    hora = models.TimeField()

    class Meta:
        managed = False
        db_table = 'log_pdv'


class LogPrecoProdutos(models.Model):
    codigo = models.AutoField(primary_key=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    precocusto = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    precovenda = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    precocontabil = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    vendedor = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='vendedor', blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'log_preco_produtos'


class LogRetornoRemessa(models.Model):
    retorno = models.ForeignKey(CabecalhoRetornoVendaFora, models.DO_NOTHING, db_column='retorno', blank=True, null=True)
    remessa = models.ForeignKey(CabecalhoRemessaVendaFora, models.DO_NOTHING, db_column='remessa', blank=True, null=True)
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'log_retorno_remessa'


class LogTipoHorasOs(models.Model):
    cd_item = models.OneToOneField('ServicosIntervalo', models.DO_NOTHING, db_column='cd_item', primary_key=True)
    valor_antigo = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    valor_novo = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    data = models.DateField()
    hora = models.TimeField()

    class Meta:
        managed = False
        db_table = 'log_tipo_horas_os'


class LogVendedorMetas(models.Model):
    codigo = models.AutoField(primary_key=True)
    codigo_vendedor_meta = models.IntegerField(blank=True, null=True)
    codigo_meta = models.IntegerField(blank=True, null=True)
    codigo_vendedor = models.IntegerField(blank=True, null=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    operacao = models.CharField(max_length=1, blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    data_hora = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'log_vendedor_metas'


class Lotes(models.Model):
    codigo = models.AutoField(primary_key=True)
    data_operacao = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)
    fornecedor = models.ForeignKey(Fornecedores, models.DO_NOTHING, db_column='fornecedor', blank=True, null=True)
    numero = models.CharField(max_length=20)
    data_fabricacao = models.DateField()
    data_validade = models.DateField()
    preco_maximo = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    codigo_agragacao = models.CharField(max_length=20, blank=True, null=True)
    descricao = models.CharField(max_length=50, blank=True, null=True)
    controla_vencimento = models.IntegerField(blank=True, null=True)
    origem = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'lotes'


class LotesFabricacaoAlterado(models.Model):
    codigo = models.AutoField(primary_key=True)
    data = models.DateTimeField()
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    lote = models.ForeignKey(Lotes, models.DO_NOTHING, db_column='lote', blank=True, null=True)
    fabricacao_atual = models.DateField(blank=True, null=True)
    fabricacao_anterior = models.DateField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'lotes_fabricacao_alterado'


class LotesMovimento(models.Model):
    codigo = models.AutoField(primary_key=True)
    data_operacao = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)
    lote = models.ForeignKey(Lotes, models.DO_NOTHING, db_column='lote')
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto')
    item_ordem_servico = models.ForeignKey(ItensOrdemServico, models.DO_NOTHING, db_column='item_ordem_servico', blank=True, null=True)
    item_ordem_servico_despesas = models.ForeignKey(CabecalhoOrdemServicoDespesas, models.DO_NOTHING, db_column='item_ordem_servico_despesas', blank=True, null=True)
    item_orcamento = models.ForeignKey(ItensOrcamento, models.DO_NOTHING, db_column='item_orcamento', blank=True, null=True)
    item_transferencia = models.ForeignKey(ItensTransferencia, models.DO_NOTHING, db_column='item_transferencia', blank=True, null=True)
    item_transferencia_entrada = models.ForeignKey(ItensTransferenciaEntrada, models.DO_NOTHING, db_column='item_transferencia_entrada', blank=True, null=True)
    item_remessa_conserto = models.ForeignKey(ItensRemessa, models.DO_NOTHING, db_column='item_remessa_conserto', blank=True, null=True)
    item_remessa_conserto_entrada = models.ForeignKey(ItensRemessaEntrada, models.DO_NOTHING, db_column='item_remessa_conserto_entrada', blank=True, null=True)
    item_devolucao_fornecedor = models.ForeignKey(ItensRemessa, models.DO_NOTHING, db_column='item_devolucao_fornecedor', blank=True, null=True)
    item_remessa_venda_fora = models.ForeignKey(ItensRemessaVendaFora, models.DO_NOTHING, db_column='item_remessa_venda_fora', blank=True, null=True)
    item_retorno_venda_fora = models.ForeignKey(ItensRetornoVendaFora, models.DO_NOTHING, db_column='item_retorno_venda_fora', blank=True, null=True)
    item_compra = models.ForeignKey(ItensNotaCompra, models.DO_NOTHING, db_column='item_compra', blank=True, null=True)
    item_avulsa_saida = models.ForeignKey(ItensNotaAvulsa, models.DO_NOTHING, db_column='item_avulsa_saida', blank=True, null=True)
    item_avulsa_entrada = models.ForeignKey(ItensNotaAvulsa, models.DO_NOTHING, db_column='item_avulsa_entrada', blank=True, null=True)
    item_devolucao = models.ForeignKey(ItensDevolucao, models.DO_NOTHING, db_column='item_devolucao', blank=True, null=True)
    item_futura = models.ForeignKey(ItensEntrega, models.DO_NOTHING, db_column='item_futura', blank=True, null=True)
    item_formula = models.ForeignKey(ItensFormula, models.DO_NOTHING, db_column='item_formula', blank=True, null=True)
    item_perda = models.ForeignKey(ItensPerda, models.DO_NOTHING, db_column='item_perda', blank=True, null=True)
    item_entrada_saida_manual = models.ForeignKey(EntradaSaidaManual, models.DO_NOTHING, db_column='item_entrada_saida_manual', blank=True, null=True)
    item_producao = models.ForeignKey(ItensProducao, models.DO_NOTHING, db_column='item_producao', blank=True, null=True)
    item_emprestei = models.ForeignKey('ProdutosEmprestei', models.DO_NOTHING, db_column='item_emprestei', blank=True, null=True)
    item_emprestei_devolvi = models.ForeignKey('ProdutosEmpresteiDevolvi', models.DO_NOTHING, db_column='item_emprestei_devolvi', blank=True, null=True)
    item_emprestado = models.ForeignKey('ProdutosEmprestados', models.DO_NOTHING, db_column='item_emprestado', blank=True, null=True)
    item_emprestado_devolvido = models.ForeignKey('ProdutosEmprestadosDevolvidos', models.DO_NOTHING, db_column='item_emprestado_devolvido', blank=True, null=True)
    item_composicao = models.ForeignKey(ComposicaoMovimento, models.DO_NOTHING, db_column='item_composicao', blank=True, null=True)
    item_composicao_venda = models.ForeignKey(ComposicaoMovimento, models.DO_NOTHING, db_column='item_composicao_venda', blank=True, null=True)
    item_composicao_futura = models.ForeignKey(ComposicaoMovimento, models.DO_NOTHING, db_column='item_composicao_futura', blank=True, null=True)
    item_lote_movimento_venda = models.ForeignKey('self', models.DO_NOTHING, db_column='item_lote_movimento_venda', blank=True, null=True)
    item_lote_movimento_futura = models.ForeignKey('self', models.DO_NOTHING, db_column='item_lote_movimento_futura', blank=True, null=True)
    item_conferencia_estoque = models.ForeignKey(ItensConferenciaEstoque, models.DO_NOTHING, db_column='item_conferencia_estoque', blank=True, null=True)
    item_produto_saida = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='item_produto_saida', blank=True, null=True)
    item_produto_entrada = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='item_produto_entrada', blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535)
    calcula_saldo = models.BooleanField()
    operacao = models.CharField(max_length=30)
    conversao = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'lotes_movimento'


class LotesMovimentoLog(models.Model):
    codigo = models.AutoField(primary_key=True)
    data_operacao = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)
    lote = models.IntegerField()
    produto = models.IntegerField()
    item_ordem_servico = models.IntegerField(blank=True, null=True)
    item_ordem_servico_despesas = models.IntegerField(blank=True, null=True)
    item_orcamento = models.IntegerField(blank=True, null=True)
    item_transferencia = models.IntegerField(blank=True, null=True)
    item_transferencia_entrada = models.IntegerField(blank=True, null=True)
    item_remessa_conserto = models.IntegerField(blank=True, null=True)
    item_remessa_conserto_entrada = models.IntegerField(blank=True, null=True)
    item_devolucao_fornecedor = models.IntegerField(blank=True, null=True)
    item_remessa_venda_fora = models.IntegerField(blank=True, null=True)
    item_retorno_venda_fora = models.IntegerField(blank=True, null=True)
    item_compra = models.IntegerField(blank=True, null=True)
    item_avulsa_saida = models.IntegerField(blank=True, null=True)
    item_avulsa_entrada = models.IntegerField(blank=True, null=True)
    item_devolucao = models.IntegerField(blank=True, null=True)
    item_futura = models.IntegerField(blank=True, null=True)
    item_formula = models.IntegerField(blank=True, null=True)
    item_perda = models.IntegerField(blank=True, null=True)
    item_entrada_saida_manual = models.IntegerField(blank=True, null=True)
    item_producao = models.IntegerField(blank=True, null=True)
    item_emprestei = models.IntegerField(blank=True, null=True)
    item_emprestei_devolvi = models.IntegerField(blank=True, null=True)
    item_emprestado = models.IntegerField(blank=True, null=True)
    item_emprestado_devolvido = models.IntegerField(blank=True, null=True)
    item_composicao = models.IntegerField(blank=True, null=True)
    item_composicao_venda = models.IntegerField(blank=True, null=True)
    item_composicao_futura = models.IntegerField(blank=True, null=True)
    item_lote_movimento_venda = models.IntegerField(blank=True, null=True)
    item_lote_movimento_futura = models.IntegerField(blank=True, null=True)
    item_conferencia_estoque = models.IntegerField(blank=True, null=True)
    item_produto_saida = models.IntegerField(blank=True, null=True)
    item_produto_entrada = models.IntegerField(blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535)
    calcula_saldo = models.BooleanField()
    operacao = models.CharField(max_length=30)
    tipo = models.CharField(max_length=10)

    class Meta:
        managed = False
        db_table = 'lotes_movimento_log'


class LotesVencimentoAlterado(models.Model):
    codigo = models.AutoField(primary_key=True)
    data = models.DateTimeField()
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    lote = models.ForeignKey(Lotes, models.DO_NOTHING, db_column='lote', blank=True, null=True)
    vencimento_atual = models.DateField(blank=True, null=True)
    vencimento_anterior = models.DateField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'lotes_vencimento_alterado'


class MapaEcf60A(models.Model):
    codigo = models.AutoField(primary_key=True)
    terminal = models.ForeignKey('MapaEcf60M', models.DO_NOTHING, db_column='terminal', blank=True, null=True)
    situacao = models.CharField(max_length=7, blank=True, null=True)
    total = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    md5 = models.CharField(max_length=-1, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'mapa_ecf_60a'


class MapaEcf60M(models.Model):
    codigo = models.AutoField(primary_key=True)
    serie = models.CharField(max_length=22, blank=True, null=True)
    terminal = models.ForeignKey('Terminais', models.DO_NOTHING, db_column='terminal', blank=True, null=True)
    movimento = models.DateField(blank=True, null=True)
    cooi = models.IntegerField(blank=True, null=True)
    coof = models.IntegerField(blank=True, null=True)
    gti = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    gtf = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    reducoes = models.IntegerField(blank=True, null=True)
    reinicios = models.IntegerField(blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)
    exportou = models.BooleanField(blank=True, null=True)
    md5 = models.CharField(max_length=-1, blank=True, null=True)
    enviou_sefaz = models.BooleanField()
    recibo = models.CharField(max_length=100, blank=True, null=True)
    dh_envio = models.DateTimeField(blank=True, null=True)
    mensagem_retorno = models.CharField(max_length=1000, blank=True, null=True)
    retorno = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'mapa_ecf_60m'


class MapaEcfFormasPagamento(models.Model):
    codigo = models.AutoField(primary_key=True)
    numero_serie = models.CharField(max_length=-1, blank=True, null=True)
    indice = models.IntegerField(blank=True, null=True)
    descricao = models.CharField(max_length=-1, blank=True, null=True)
    tipo_documento = models.CharField(max_length=-1, blank=True, null=True)
    valor_acumulado = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    data_acumulacao = models.DateField(blank=True, null=True)
    soma_individual = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    ordem = models.ForeignKey(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', blank=True, null=True)
    ccf = models.IntegerField(blank=True, null=True)
    coo = models.IntegerField(blank=True, null=True)
    gnf = models.IntegerField(blank=True, null=True)
    estornado = models.BooleanField(blank=True, null=True)
    exportou = models.BooleanField(blank=True, null=True)
    md5 = models.CharField(max_length=-1)
    credito = models.BooleanField(blank=True, null=True)
    debito = models.BooleanField(blank=True, null=True)
    md5_a2 = models.CharField(max_length=-1)

    class Meta:
        managed = False
        db_table = 'mapa_ecf_formas_pagamento'


class Marcas(models.Model):
    codigo = models.AutoField(primary_key=True)
    nome = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'marcas'


class Md5Dav(models.Model):
    ordem = models.ForeignKey(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', blank=True, null=True)
    md5 = models.CharField(max_length=-1, blank=True, null=True)
    coo = models.IntegerField(blank=True, null=True)
    codigo = models.AutoField(primary_key=True)
    numero_serie_impressora = models.CharField(max_length=-1, blank=True, null=True)
    orcamento = models.ForeignKey(CabecalhoOrcamento, models.DO_NOTHING, db_column='orcamento', blank=True, null=True)
    data_inclusao = models.DateField()
    numero_dav = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'md5_dav'


class MemoriaFiscalImpressa(models.Model):
    numero_serie = models.CharField(primary_key=True, max_length=-1)
    data_movimento = models.CharField(max_length=-1)

    class Meta:
        managed = False
        db_table = 'memoria_fiscal_impressa'
        unique_together = (('numero_serie', 'data_movimento'),)


class Mensagemnf(models.Model):
    codigo = models.AutoField(primary_key=True)
    descricao = models.TextField()

    class Meta:
        managed = False
        db_table = 'mensagemnf'


class Mensagens(models.Model):
    codigo = models.DecimalField(primary_key=True, max_digits=65535, decimal_places=65535)
    nome = models.CharField(max_length=100, blank=True, null=True)
    ativo = models.BooleanField()
    mensagem1 = models.CharField(max_length=60, blank=True, null=True)
    mensagem2 = models.CharField(max_length=60, blank=True, null=True)
    mensagem3 = models.CharField(max_length=60, blank=True, null=True)
    mensagem4 = models.CharField(max_length=60, blank=True, null=True)
    mensagem5 = models.CharField(max_length=60, blank=True, null=True)
    mensagem6 = models.CharField(max_length=60, blank=True, null=True)
    mensagem7 = models.CharField(max_length=60, blank=True, null=True)
    mensagem8 = models.CharField(max_length=60, blank=True, null=True)
    mensagem9 = models.CharField(max_length=60, blank=True, null=True)
    mensagem10 = models.CharField(max_length=60, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'mensagens'


class MensagensRecebidas(models.Model):
    codigo = models.IntegerField(primary_key=True)
    mensagem = models.TextField()
    visualizada = models.BooleanField(blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    datahora = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'mensagens_recebidas'


class MfePagamentos(models.Model):
    codigo = models.AutoField(primary_key=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    datahora = models.DateTimeField()
    ordem = models.ForeignKey(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem')
    idfila = models.IntegerField()
    idpagamentolocal = models.IntegerField()
    bandeira = models.CharField(max_length=100)
    adquirente = models.CharField(max_length=100)
    nsu = models.CharField(max_length=100)
    codigo_autorizacao = models.CharField(max_length=100)
    bin = models.CharField(max_length=6)
    ultimos_quatro_digitos = models.CharField(max_length=4)
    dono_cartao = models.CharField(max_length=100)
    data_expiracao = models.CharField(max_length=7)
    enviado_sefaz = models.BooleanField()
    cancelado = models.BooleanField()
    id_resposta_fiscal = models.CharField(max_length=-1)
    equipamento_pos = models.ForeignKey(EquipamentosPos, models.DO_NOTHING, db_column='equipamento_pos', blank=True, null=True)
    numero_sessao = models.IntegerField()
    valor = models.DecimalField(max_digits=65535, decimal_places=65535)

    class Meta:
        managed = False
        db_table = 'mfe_pagamentos'


class ModeloBoletos(models.Model):
    codigo = models.AutoField(primary_key=True)
    modelo = models.IntegerField(blank=True, null=True)
    linha_inicial = models.IntegerField(blank=True, null=True)
    linha_entre = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'modelo_boletos'


class Modelonf(models.Model):
    codigo = models.IntegerField(primary_key=True)
    descricao = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'modelonf'


class Modelos(models.Model):
    codigo = models.AutoField(primary_key=True)
    nome = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'modelos'


class ModelosNf(models.Model):
    modelo = models.CharField(primary_key=True, max_length=2)
    descricao = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'modelos_nf'


class ModuloSistema(models.Model):
    codigo = models.AutoField(primary_key=True)
    nome = models.CharField(max_length=50)
    data_hora = models.DateTimeField(blank=True, null=True)
    usuario = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'modulo_sistema'


class MontagemMotor(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nome = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'montagem_motor'


class MotivoCancelamentoNfse(models.Model):
    codigo = models.CharField(primary_key=True, max_length=7)
    descricao = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'motivo_cancelamento_nfse'


class Motivos(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nome = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'motivos'


class MovSeriais(models.Model):
    codigo = models.AutoField(primary_key=True)
    serial = models.ForeignKey('Seriais', models.DO_NOTHING, db_column='serial', blank=True, null=True)
    tipo = models.CharField(max_length=20, blank=True, null=True)
    controle = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'mov_seriais'


class MovimentoPorEcf(models.Model):
    codigo = models.AutoField(primary_key=True)
    numero_serie = models.CharField(max_length=-1)
    datainicial = models.DateField()
    datafinal = models.DateField()
    md5 = models.CharField(max_length=-1, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'movimento_por_ecf'


class NfeCanceladas(models.Model):
    chave_nota = models.OneToOneField('NotasEletronicas', models.DO_NOTHING, db_column='chave_nota', primary_key=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    motivo = models.ForeignKey(Motivos, models.DO_NOTHING, db_column='motivo', blank=True, null=True)
    complemento = models.CharField(max_length=100, blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'nfe_canceladas'


class NfeNaoCancelada(models.Model):
    chave_nota = models.OneToOneField('NotasEletronicas', models.DO_NOTHING, db_column='chave_nota', primary_key=True)
    id = models.CharField(max_length=-1)

    class Meta:
        managed = False
        db_table = 'nfe_nao_cancelada'
        unique_together = (('chave_nota', 'id'),)


class NfseCanceladas(models.Model):
    codigo = models.AutoField(primary_key=True)
    motivo = models.ForeignKey(MotivoCancelamentoNfse, models.DO_NOTHING, db_column='motivo')
    complemento = models.CharField(max_length=-1, blank=True, null=True)
    chave_nota = models.ForeignKey('NotasEletronicas', models.DO_NOTHING, db_column='chave_nota')
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    data = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'nfse_canceladas'


class NiveisDeAcesso(models.Model):
    codigo = models.AutoField(primary_key=True)
    descricao = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'niveis_de_acesso'


class NiveisDeAcessoPermissoes(models.Model):
    codigo_nivel = models.ForeignKey(NiveisDeAcesso, models.DO_NOTHING, db_column='codigo_nivel', blank=True, null=True)
    opcao_sistema = models.IntegerField(blank=True, null=True)
    nivel = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'niveis_de_acesso_permissoes'


class NoFornecedor(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_fornecedor = models.ForeignKey(Fornecedores, models.DO_NOTHING, db_column='cd_fornecedor', blank=True, null=True)
    cd_produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='cd_produto', blank=True, null=True)
    no_fornecedor = models.CharField(max_length=20, blank=True, null=True)
    fracao = models.DecimalField(max_digits=20, decimal_places=6)

    class Meta:
        managed = False
        db_table = 'no_fornecedor'


class NotaCompraXml(models.Model):
    nota = models.OneToOneField(CabecalhoNotaCompra, models.DO_NOTHING, db_column='nota', primary_key=True)
    xml = models.TextField(blank=True, null=True)
    enviado = models.BooleanField(blank=True, null=True)
    naoenviar = models.BooleanField(blank=True, null=True)
    reimportada = models.BooleanField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'nota_compra_xml'


class Notas(models.Model):
    nota = models.AutoField()
    ordem = models.IntegerField(blank=True, null=True)
    cfop = models.ForeignKey(Cfop, models.DO_NOTHING, db_column='cfop', blank=True, null=True)
    dataemissao = models.DateField(blank=True, null=True)
    cancelada = models.CharField(max_length=1, blank=True, null=True)
    valor_produto = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    total_servicos = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    valor_total_nota = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    frete = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    seguro = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    outras_despesas = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    valor_iss = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    base_icms = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    valor_icms = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    transportadora = models.ForeignKey(Fornecedores, models.DO_NOTHING, db_column='transportadora', blank=True, null=True)
    frete_por_conta = models.IntegerField(blank=True, null=True)
    placa = models.CharField(max_length=10, blank=True, null=True)
    uf = models.CharField(max_length=2, blank=True, null=True)
    icm_subst = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    valor_subst = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    quantidade = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    especie = models.CharField(max_length=20, blank=True, null=True)
    marca = models.CharField(max_length=20, blank=True, null=True)
    numero = models.CharField(max_length=15, blank=True, null=True)
    peso_bruto = models.DecimalField(max_digits=12, decimal_places=4, blank=True, null=True)
    peso_liquido = models.DecimalField(max_digits=12, decimal_places=4, blank=True, null=True)
    redesp_transp = models.ForeignKey(Fornecedores, models.DO_NOTHING, db_column='redesp_transp', blank=True, null=True)
    redesp_placa = models.CharField(max_length=10, blank=True, null=True)
    modelo = models.ForeignKey(Conffiscal, models.DO_NOTHING, db_column='modelo', blank=True, null=True)
    data_saida = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)
    base_iss = models.DecimalField(max_digits=12, decimal_places=4, blank=True, null=True)
    controle_hora = models.TimeField(blank=True, null=True)
    controle_data = models.DateField(blank=True, null=True)
    serie = models.CharField(max_length=3, blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)
    tipo = models.CharField(max_length=30, blank=True, null=True)
    codigo = models.AutoField(primary_key=True)
    nsu = models.AutoField()
    total_ipi = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    observacoes = models.CharField(max_length=-1, blank=True, null=True)
    cod_obs = models.ForeignKey('ObservacoesMovimentos', models.DO_NOTHING, db_column='cod_obs', blank=True, null=True)
    desconto_valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    acrescimo_valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    total_pis = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    total_cofins = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    total_pis_st = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    total_cofins_st = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    subserie = models.IntegerField()
    tipo_emissao = models.CharField(max_length=20, blank=True, null=True)
    cep = models.ForeignKey(Cidades, models.DO_NOTHING, db_column='cep', blank=True, null=True)
    cfop_transporte = models.ForeignKey(Cfop, models.DO_NOTHING, db_column='cfop_transporte', blank=True, null=True)
    modelo_nota = models.CharField(max_length=4)
    informacoesfisco = models.CharField(max_length=4000, blank=True, null=True)
    cabecalho_ordem_servico = models.ForeignKey(CabecalhoOrdemServico, models.DO_NOTHING, db_column='cabecalho_ordem_servico', blank=True, null=True)
    cabeca_nota_avulsa = models.ForeignKey(CabecaNotaAvulsa, models.DO_NOTHING, db_column='cabeca_nota_avulsa', blank=True, null=True)
    parcelas = models.ForeignKey('Parcelas', models.DO_NOTHING, db_column='parcelas', blank=True, null=True)
    cabecalho_devolucao = models.ForeignKey(CabecalhoDevolucao, models.DO_NOTHING, db_column='cabecalho_devolucao', blank=True, null=True)
    remessa_conserto = models.ForeignKey('Remessa', models.DO_NOTHING, db_column='remessa_conserto', blank=True, null=True)
    devolucao_fornecedor = models.ForeignKey('Remessa', models.DO_NOTHING, db_column='devolucao_fornecedor', blank=True, null=True)
    transferencia = models.ForeignKey('Transferencia', models.DO_NOTHING, db_column='transferencia', blank=True, null=True)
    cabecalho_remessa_venda_fora = models.ForeignKey(CabecalhoRemessaVendaFora, models.DO_NOTHING, db_column='cabecalho_remessa_venda_fora', blank=True, null=True)
    cabecalho_retorno_venda_fora = models.ForeignKey(CabecalhoRetornoVendaFora, models.DO_NOTHING, db_column='cabecalho_retorno_venda_fora', blank=True, null=True)
    ordem_producao = models.ForeignKey('OrdemProducao', models.DO_NOTHING, db_column='ordem_producao', blank=True, null=True)
    cabecalho_nota_compra = models.ForeignKey(CabecalhoNotaCompra, models.DO_NOTHING, db_column='cabecalho_nota_compra', blank=True, null=True)
    cabecalho_entrega = models.ForeignKey(CabecalhoEntrega, models.DO_NOTHING, db_column='cabecalho_entrega', blank=True, null=True)
    scan = models.BooleanField()
    mei = models.BooleanField()
    total_fcp_st = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    total_fcp_icms = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente', blank=True, null=True)
    fornecedor = models.ForeignKey(Fornecedores, models.DO_NOTHING, db_column='fornecedor', blank=True, null=True)
    filial = models.ForeignKey(Filiais, models.DO_NOTHING, db_column='filial', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'notas'


class NotasEletronicas(models.Model):
    chave_nota = models.OneToOneField(Notas, models.DO_NOTHING, db_column='chave_nota', primary_key=True)
    id = models.CharField(max_length=50, blank=True, null=True)
    numero_nota = models.IntegerField(blank=True, null=True)
    serie = models.CharField(max_length=3, blank=True, null=True)
    certificado = models.CharField(max_length=50, blank=True, null=True)
    ambiente = models.SmallIntegerField(blank=True, null=True)
    uf_responsavel = models.CharField(max_length=2, blank=True, null=True)
    status = models.CharField(max_length=20)
    motivo = models.CharField(max_length=5000, blank=True, null=True)
    lote = models.IntegerField(blank=True, null=True)
    tempo_espera = models.IntegerField(blank=True, null=True)
    dh_recibo = models.CharField(max_length=50, blank=True, null=True)
    dh_recibo_cancelamento = models.CharField(max_length=50, blank=True, null=True)
    recibo = models.CharField(max_length=50, blank=True, null=True)
    protocolo = models.CharField(max_length=50, blank=True, null=True)
    enviada = models.BooleanField(blank=True, null=True)
    impressa = models.BooleanField(blank=True, null=True)
    marcada = models.CharField(max_length=1, blank=True, null=True)
    cnpj_destinatario = models.CharField(max_length=18, blank=True, null=True)
    agrupadas = models.BooleanField(blank=True, null=True)
    versao_dados = models.CharField(max_length=4)
    versao_servico = models.CharField(max_length=4)
    endereco_consulta = models.CharField(max_length=1000)
    codigo_retorno = models.IntegerField()
    protocolodecancelamentoextemporaneo = models.CharField(max_length=100)
    erro_transmitir = models.BooleanField()
    id_cancelamento_sat = models.CharField(max_length=100, blank=True, null=True)
    assinaturaqrcode_cancelamento = models.CharField(max_length=500, blank=True, null=True)
    assinaturaqrcode = models.CharField(max_length=500, blank=True, null=True)
    sincrono = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'notas_eletronicas'


class NotasEletronicasContingencia(models.Model):
    chave_nota = models.OneToOneField(NotasEletronicas, models.DO_NOTHING, db_column='chave_nota', primary_key=True)
    motivo_contingencia = models.CharField(max_length=500)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    autorizou_apos_prazo = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'notas_eletronicas_contingencia'


class NotasEletronicasEmail(models.Model):
    chave_nota = models.OneToOneField(NotasEletronicas, models.DO_NOTHING, db_column='chave_nota', primary_key=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    data = models.DateField()
    hora = models.TimeField()

    class Meta:
        managed = False
        db_table = 'notas_eletronicas_email'


class NotasEletronicasEmailTransportadora(models.Model):
    chave_nota = models.OneToOneField(NotasEletronicas, models.DO_NOTHING, db_column='chave_nota', primary_key=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    data = models.DateField()
    hora = models.TimeField()

    class Meta:
        managed = False
        db_table = 'notas_eletronicas_email_transportadora'


class NotasEletronicasRejeitadas(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    pedido = models.IntegerField()
    operacao = models.CharField(max_length=30)
    chave = models.CharField(max_length=44)
    recibo = models.CharField(max_length=50)
    motivo = models.CharField(max_length=10000)
    consultada = models.BooleanField()
    versao_dados = models.CharField(max_length=10)
    certificado = models.CharField(max_length=50)
    xml = models.TextField()
    nota = models.IntegerField()
    serie = models.CharField(max_length=3)
    modelo = models.CharField(max_length=2)
    ambiente = models.IntegerField()
    sincrono = models.BooleanField()
    cfop = models.ForeignKey(Cfop, models.DO_NOTHING, db_column='cfop')
    scan = models.BooleanField()
    endereco_consulta = models.CharField(max_length=10000)
    lote = models.IntegerField()
    cfop_transporte = models.ForeignKey(Cfop, models.DO_NOTHING, db_column='cfop_transporte', blank=True, null=True)
    dataemissao = models.DateField()
    erro_consulta = models.CharField(max_length=10000)

    class Meta:
        managed = False
        db_table = 'notas_eletronicas_rejeitadas'


class NotasEletronicasXml(models.Model):
    chave_nota = models.OneToOneField(Notas, models.DO_NOTHING, db_column='chave_nota', primary_key=True)
    xml = models.TextField(blank=True, null=True)
    enviado = models.BooleanField()
    nota = models.TextField(blank=True, null=True)
    xml_cancelamento = models.TextField(blank=True, null=True)
    nfe_pdf = models.BinaryField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'notas_eletronicas_xml'


class NotasInutilizadas(models.Model):
    chave_nota = models.OneToOneField(Notas, models.DO_NOTHING, db_column='chave_nota', primary_key=True)
    ambiente = models.IntegerField(blank=True, null=True)
    status = models.CharField(max_length=-1, blank=True, null=True)
    nf_inicial = models.IntegerField(blank=True, null=True)
    nf_final = models.IntegerField(blank=True, null=True)
    dhrecibo = models.CharField(max_length=-1, blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    datahora = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'notas_inutilizadas'


class NotasLancamento(models.Model):
    chave_nota = models.OneToOneField(Notas, models.DO_NOTHING, db_column='chave_nota', primary_key=True)

    class Meta:
        managed = False
        db_table = 'notas_lancamento'


class NotasMd5J1(models.Model):
    cd_nota = models.ForeignKey(Notas, models.DO_NOTHING, db_column='cd_nota', blank=True, null=True)
    md5 = models.CharField(max_length=32, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'notas_md5_j1'


class NotasNfese(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_nota = models.ForeignKey(Notas, models.DO_NOTHING, db_column='cd_nota', blank=True, null=True)
    rps = models.IntegerField(blank=True, null=True)
    provedor = models.CharField(max_length=30, blank=True, null=True)
    codigo_verificacao = models.CharField(max_length=50, blank=True, null=True)
    endereco_impressao_nfese = models.TextField(blank=True, null=True)
    xml_rps = models.TextField(blank=True, null=True)
    naturezaoperacao = models.CharField(max_length=5, blank=True, null=True)
    cidadeservico = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'notas_nfese'


class NotasOrdenacaoItens(models.Model):
    chave_nota = models.OneToOneField(Notas, models.DO_NOTHING, db_column='chave_nota', primary_key=True)
    ordenacao = models.CharField(max_length=-1, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'notas_ordenacao_itens'


class NotasParcelas(models.Model):
    parcela = models.OneToOneField('Parcelas', models.DO_NOTHING, db_column='parcela', primary_key=True)
    nota = models.ForeignKey(Notas, models.DO_NOTHING, db_column='nota')

    class Meta:
        managed = False
        db_table = 'notas_parcelas'
        unique_together = (('parcela', 'nota'),)


class NsuAlteracoes(models.Model):
    codigo = models.AutoField(primary_key=True)
    numero_anterior = models.IntegerField(blank=True, null=True)
    novo_numero = models.IntegerField(blank=True, null=True)
    motivo = models.CharField(max_length=50, blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'nsu_alteracoes'


class ObjetosServicos(models.Model):
    codigo = models.AutoField(primary_key=True)
    cod_servicos = models.ForeignKey('Servicos', models.DO_NOTHING, db_column='cod_servicos', blank=True, null=True)
    objeto = models.CharField(max_length=100, blank=True, null=True)
    problema = models.CharField(max_length=5000, blank=True, null=True)
    tecnico = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='tecnico', blank=True, null=True)
    marca = models.CharField(max_length=50, blank=True, null=True)
    modelo = models.CharField(max_length=50, blank=True, null=True)
    serie = models.CharField(max_length=20, blank=True, null=True)
    cor = models.CharField(max_length=20, blank=True, null=True)
    patrimonio = models.CharField(max_length=20, blank=True, null=True)
    id = models.CharField(max_length=20, blank=True, null=True)
    constatado = models.CharField(max_length=5000, blank=True, null=True)
    ano = models.CharField(max_length=9, blank=True, null=True)
    combustivel = models.CharField(max_length=20, blank=True, null=True)
    km = models.CharField(max_length=20, blank=True, null=True)
    hora_recebida = models.TimeField(blank=True, null=True)
    hora_entrega = models.TimeField(blank=True, null=True)
    situacao = models.CharField(max_length=20, blank=True, null=True)
    data_entrega = models.DateField(blank=True, null=True)
    responsavel = models.CharField(max_length=100, blank=True, null=True)
    linha = models.CharField(max_length=100, blank=True, null=True)
    chassi = models.CharField(max_length=50, blank=True, null=True)
    nivel_combustivel = models.CharField(max_length=50, blank=True, null=True)
    acessorios = models.CharField(max_length=1000, blank=True, null=True)
    renavam = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    situacao_anterior = models.CharField(max_length=20)

    class Meta:
        managed = False
        db_table = 'objetos_servicos'


class ObjetosServicosDadosEquipamentos(models.Model):
    objeto = models.OneToOneField(ObjetosServicos, models.DO_NOTHING, db_column='objeto', primary_key=True)
    aplicacao = models.CharField(max_length=50, blank=True, null=True)
    codigo = models.CharField(max_length=15, blank=True, null=True)
    setor = models.CharField(max_length=15, blank=True, null=True)
    responsavel = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'objetos_servicos_dados_equipamentos'


class ObjetosServicosDadosTecnicos(models.Model):
    objeto = models.OneToOneField(ObjetosServicos, models.DO_NOTHING, db_column='objeto', primary_key=True)
    passo_polar = models.CharField(max_length=30, blank=True, null=True)
    pacote = models.CharField(max_length=20, blank=True, null=True)
    espira = models.CharField(max_length=30, blank=True, null=True)
    diamento = models.CharField(max_length=25, blank=True, null=True)
    fio = models.CharField(max_length=25, blank=True, null=True)
    camada = models.CharField(max_length=25, blank=True, null=True)
    bobina = models.CharField(max_length=25, blank=True, null=True)
    resistencia = models.CharField(max_length=25, blank=True, null=True)
    io = models.CharField(max_length=25, blank=True, null=True)
    grupo = models.CharField(max_length=25, blank=True, null=True)
    potencia_vazio = models.CharField(max_length=25, blank=True, null=True)
    ranhuras = models.CharField(max_length=25, blank=True, null=True)
    ligacao = models.CharField(max_length=25, blank=True, null=True)
    esquema = models.CharField(max_length=25, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'objetos_servicos_dados_tecnicos'


class ObservaItensProducao(models.Model):
    cd_item = models.OneToOneField(ItensProducao, models.DO_NOTHING, db_column='cd_item', primary_key=True)
    cd_obs = models.ForeignKey('ObservacoesMovimentos', models.DO_NOTHING, db_column='cd_obs', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'observa_itens_producao'


class ObservaItensVenda(models.Model):
    cd_item = models.OneToOneField(ItensOrdemServico, models.DO_NOTHING, db_column='cd_item', primary_key=True)
    cd_obs = models.ForeignKey('ObservacoesMovimentos', models.DO_NOTHING, db_column='cd_obs', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'observa_itens_venda'


class ObservaServicos(models.Model):
    cd_os = models.OneToOneField('Servicos', models.DO_NOTHING, db_column='cd_os', primary_key=True)
    cd_obs = models.ForeignKey('ObservacoesMovimentos', models.DO_NOTHING, db_column='cd_obs', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'observa_servicos'


class Observacoes(models.Model):
    codigo = models.AutoField(primary_key=True)
    descricao = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'observacoes'


class ObservacoesMovimentos(models.Model):
    codigo = models.AutoField(primary_key=True)
    descricao = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'observacoes_movimentos'


class OrcamentoProcedimentos(models.Model):
    orcamento = models.OneToOneField(CabecalhoOrcamento, models.DO_NOTHING, db_column='orcamento', primary_key=True)
    procedimento = models.ForeignKey('Procedimentos', models.DO_NOTHING, db_column='procedimento', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'orcamento_procedimentos'


class OrdemProducao(models.Model):
    ordem = models.AutoField(primary_key=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador', blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)
    dataf = models.DateField(blank=True, null=True)
    horaf = models.TimeField(blank=True, null=True)
    status = models.CharField(max_length=1)
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente', blank=True, null=True)
    observacao = models.ForeignKey(ObservacoesMovimentos, models.DO_NOTHING, db_column='observacao', blank=True, null=True)
    data_cancelamento = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'ordem_producao'


class OrdemProducaoImportadas(models.Model):
    ordem = models.OneToOneField(OrdemProducao, models.DO_NOTHING, db_column='ordem', primary_key=True)

    class Meta:
        managed = False
        db_table = 'ordem_producao_importadas'


class OrdemServicoProducao(models.Model):
    codigo = models.AutoField(primary_key=True)
    item = models.ForeignKey('OsServicos', models.DO_NOTHING, db_column='item', blank=True, null=True)
    producao = models.ForeignKey(OrdemProducao, models.DO_NOTHING, db_column='producao', blank=True, null=True)
    data_hora = models.DateTimeField(blank=True, null=True)
    operador = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'ordem_servico_producao'


class OsAutorizada(models.Model):
    os = models.OneToOneField('Servicos', models.DO_NOTHING, db_column='os', primary_key=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)
    data = models.DateField()

    class Meta:
        managed = False
        db_table = 'os_autorizada'


class OsOrcamento(models.Model):
    cd_objeto = models.OneToOneField(ObjetosServicos, models.DO_NOTHING, db_column='cd_objeto', primary_key=True)
    orcamento = models.ForeignKey(CabecalhoOrcamento, models.DO_NOTHING, db_column='orcamento')

    class Meta:
        managed = False
        db_table = 'os_orcamento'
        unique_together = (('cd_objeto', 'orcamento'),)


class OsServicos(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_objeto = models.ForeignKey(ObjetosServicos, models.DO_NOTHING, db_column='cd_objeto', blank=True, null=True)
    cd_produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='cd_produto', blank=True, null=True)
    cd_vendedor = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='cd_vendedor', blank=True, null=True)
    descricao = models.CharField(max_length=100, blank=True, null=True)
    unitario = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    comissao = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    iss = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    precocusto = models.DecimalField(max_digits=65535, decimal_places=65535)
    codbarra = models.CharField(max_length=14)
    lancou_codigo_interno = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'os_servicos'


class OsServicosPecas(models.Model):
    cd_item = models.OneToOneField(OsServicos, models.DO_NOTHING, db_column='cd_item', primary_key=True)
    desconto = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    garantia = models.CharField(max_length=50, blank=True, null=True)
    tipo_desconto = models.CharField(max_length=1, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'os_servicos_pecas'


class OsTransformada(models.Model):
    cod_pedido = models.OneToOneField(CabecalhoOrdemServico, models.DO_NOTHING, db_column='cod_pedido', primary_key=True)
    cod_ordem = models.ForeignKey('Servicos', models.DO_NOTHING, db_column='cod_ordem')

    class Meta:
        managed = False
        db_table = 'os_transformada'
        unique_together = (('cod_pedido', 'cod_ordem'),)


class PafEcfRegistroA2(models.Model):
    codigo = models.AutoField(primary_key=True)
    terminal = models.ForeignKey('Terminais', models.DO_NOTHING, db_column='terminal')
    valor = models.DecimalField(max_digits=65535, decimal_places=65535)
    tipo_documento = models.CharField(max_length=2)
    data = models.DateField()
    especie = models.ForeignKey(Especies, models.DO_NOTHING, db_column='especie')
    md5 = models.CharField(max_length=32)

    class Meta:
        managed = False
        db_table = 'paf_ecf_registro_a2'


class PafLogs(models.Model):
    codigo = models.IntegerField()
    tabela = models.CharField(max_length=-1)
    data = models.DateField()

    class Meta:
        managed = False
        db_table = 'paf_logs'
        unique_together = (('codigo', 'tabela'),)


class PagamentoNotaCompra(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_nota = models.ForeignKey(CabecalhoNotaCompra, models.DO_NOTHING, db_column='cd_nota', blank=True, null=True)
    tipo_doc = models.CharField(max_length=2, blank=True, null=True)
    documento = models.CharField(max_length=20, blank=True, null=True)
    emissao = models.DateField(blank=True, null=True)
    vencimento = models.DateField(blank=True, null=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    historico = models.ForeignKey(Historicos, models.DO_NOTHING, db_column='historico', blank=True, null=True)
    complemento = models.CharField(max_length=50, blank=True, null=True)
    portador = models.ForeignKey(Bancos, models.DO_NOTHING, db_column='portador', blank=True, null=True)
    atualizou = models.BooleanField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'pagamento_nota_compra'


class Paises(models.Model):
    codigo = models.IntegerField(primary_key=True)
    descricao = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'paises'


class PalmClientes(models.Model):
    cnpj = models.CharField(primary_key=True, max_length=18)
    razao = models.CharField(max_length=50, blank=True, null=True)
    fantasia = models.CharField(max_length=50, blank=True, null=True)
    cep = models.ForeignKey(Cidades, models.DO_NOTHING, db_column='cep', blank=True, null=True)
    endereco = models.CharField(max_length=50, blank=True, null=True)
    numero = models.CharField(max_length=10, blank=True, null=True)
    bairro = models.CharField(max_length=50, blank=True, null=True)
    complemento = models.CharField(max_length=-1, blank=True, null=True)
    fone = models.CharField(max_length=14, blank=True, null=True)
    fax = models.CharField(max_length=14, blank=True, null=True)
    celular = models.CharField(max_length=14, blank=True, null=True)
    grupo = models.ForeignKey(GrupoClientes, models.DO_NOTHING, db_column='grupo', blank=True, null=True)
    regiao = models.ForeignKey('Regioes', models.DO_NOTHING, db_column='regiao', blank=True, null=True)
    rota = models.ForeignKey('Rotas', models.DO_NOTHING, db_column='rota', blank=True, null=True)
    atividade = models.ForeignKey(Atividades, models.DO_NOTHING, db_column='atividade', blank=True, null=True)
    especie = models.ForeignKey(Especies, models.DO_NOTHING, db_column='especie', blank=True, null=True)
    condicao_pgto = models.ForeignKey('Vencimentos', models.DO_NOTHING, db_column='condicao_pgto', blank=True, null=True)
    portador = models.ForeignKey(Bancos, models.DO_NOTHING, db_column='portador', blank=True, null=True)
    ie = models.CharField(max_length=20, blank=True, null=True)
    novo = models.BooleanField(blank=True, null=True)
    lote = models.ForeignKey('PalmLote', models.DO_NOTHING, db_column='lote', blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)
    alterou = models.BooleanField(blank=True, null=True)
    tabela_preco = models.ForeignKey(CabecaPreco, models.DO_NOTHING, db_column='tabela_preco', blank=True, null=True)
    codigo = models.IntegerField(blank=True, null=True)
    fone2 = models.CharField(max_length=14, blank=True, null=True)
    email = models.CharField(max_length=50, blank=True, null=True)
    url = models.CharField(max_length=50, blank=True, null=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    data = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'palm_clientes'


class PalmClientesContato(models.Model):
    cnpj = models.OneToOneField(PalmClientes, models.DO_NOTHING, db_column='cnpj', primary_key=True)
    contato = models.CharField(max_length=50, blank=True, null=True)
    fone = models.CharField(max_length=14, blank=True, null=True)
    observacao = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'palm_clientes_contato'


class PalmClientesInfBancarias(models.Model):
    cnpj = models.OneToOneField(PalmClientes, models.DO_NOTHING, db_column='cnpj', primary_key=True)
    banco = models.IntegerField(blank=True, null=True)
    agencia = models.CharField(max_length=6, blank=True, null=True)
    conta = models.CharField(max_length=20, blank=True, null=True)
    fone = models.CharField(max_length=14, blank=True, null=True)
    contato = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'palm_clientes_inf_bancarias'


class PalmClientesRefComercial(models.Model):
    cnpj = models.OneToOneField(PalmClientes, models.DO_NOTHING, db_column='cnpj', primary_key=True)
    empresa = models.CharField(max_length=50, blank=True, null=True)
    fone = models.CharField(max_length=14, blank=True, null=True)
    contato = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'palm_clientes_ref_comercial'


class PalmEnderecoEntrega(models.Model):
    pedido = models.OneToOneField('PalmPedidos', models.DO_NOTHING, db_column='pedido', primary_key=True)
    endereco = models.CharField(max_length=100, blank=True, null=True)
    bairro = models.CharField(max_length=50, blank=True, null=True)
    cep = models.CharField(max_length=9, blank=True, null=True)
    fone = models.CharField(max_length=14, blank=True, null=True)
    complemento = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'palm_endereco_entrega'


class PalmItensPedido(models.Model):
    codigo = models.AutoField(primary_key=True)
    pedido = models.ForeignKey('PalmPedidos', models.DO_NOTHING, db_column='pedido', blank=True, null=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    descricao = models.CharField(max_length=50, blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    desconto = models.DecimalField(max_digits=65535, decimal_places=65535)
    tipo = models.CharField(max_length=1)
    qtd_devolvida = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    tabela_preco = models.ForeignKey(CabecaPreco, models.DO_NOTHING, db_column='tabela_preco', blank=True, null=True)
    bonificado = models.CharField(max_length=1, blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    data = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'palm_itens_pedido'


class PalmLote(models.Model):
    numero = models.AutoField(primary_key=True)
    data_hora = models.DateTimeField()
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')

    class Meta:
        managed = False
        db_table = 'palm_lote'


class PalmLoteErro(models.Model):
    numero = models.IntegerField(primary_key=True)
    erro = models.CharField(max_length=-1, blank=True, null=True)
    data_hora = models.DateTimeField()
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')

    class Meta:
        managed = False
        db_table = 'palm_lote_erro'


class PalmNumeroPedidoPalm(models.Model):
    pedido = models.ForeignKey('PalmPedidos', models.DO_NOTHING, db_column='pedido', blank=True, null=True)
    pedido_palm = models.IntegerField(primary_key=True)
    data = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'palm_numero_pedido_palm'


class PalmPedidoCancelados(models.Model):
    pedido = models.OneToOneField('PalmPedidos', models.DO_NOTHING, db_column='pedido', primary_key=True)
    data = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)
    complemento = models.CharField(max_length=50, blank=True, null=True)
    motivo = models.ForeignKey(Motivos, models.DO_NOTHING, db_column='motivo', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'palm_pedido_cancelados'


class PalmPedidoObservacoes(models.Model):
    pedido = models.OneToOneField('PalmPedidos', models.DO_NOTHING, db_column='pedido', primary_key=True)
    observacao = models.CharField(max_length=-1, blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    data = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'palm_pedido_observacoes'


class PalmPedidos(models.Model):
    pedido = models.AutoField(primary_key=True)
    data = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)
    cnpj = models.ForeignKey(PalmClientes, models.DO_NOTHING, db_column='cnpj', blank=True, null=True)
    condicao_pgto = models.ForeignKey('Vencimentos', models.DO_NOTHING, db_column='condicao_pgto', blank=True, null=True)
    especie = models.ForeignKey(Especies, models.DO_NOTHING, db_column='especie', blank=True, null=True)
    portador = models.ForeignKey(Bancos, models.DO_NOTHING, db_column='portador', blank=True, null=True)
    frete = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    acrescimo = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    desconto = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    marcado = models.CharField(max_length=1, blank=True, null=True)
    observacao = models.CharField(max_length=20, blank=True, null=True)
    especial = models.CharField(max_length=1, blank=True, null=True)
    lote = models.ForeignKey(PalmLote, models.DO_NOTHING, db_column='lote', blank=True, null=True)
    pedido_palm = models.IntegerField(blank=True, null=True)
    vendedor = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='vendedor', blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    data_operacao = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'palm_pedidos'


class PalmVisitaNegativa(models.Model):
    pedido = models.OneToOneField(PalmPedidos, models.DO_NOTHING, db_column='pedido', primary_key=True)
    observacao = models.CharField(max_length=200, blank=True, null=True)
    motivo = models.ForeignKey(Motivos, models.DO_NOTHING, db_column='motivo', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'palm_visita_negativa'


class Parametros(models.Model):
    descricao = models.CharField(primary_key=True, max_length=-1)
    valor = models.CharField(max_length=-1, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'parametros'


class ParcelamentoTefEmitirNota(models.Model):
    codigo = models.AutoField(primary_key=True)
    rede = models.CharField(max_length=50, blank=True, null=True)
    operacao = models.IntegerField(blank=True, null=True)
    especie = models.IntegerField(blank=True, null=True)
    parcelas = models.IntegerField(blank=True, null=True)
    total = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    identificacao = models.CharField(max_length=50, blank=True, null=True)
    pid = models.BigIntegerField()

    class Meta:
        managed = False
        db_table = 'parcelamento_tef_emitir_nota'


class Parcelas(models.Model):
    ordem = models.ForeignKey(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', blank=True, null=True)
    parcela = models.IntegerField(blank=True, null=True)
    valor = models.DecimalField(max_digits=12, decimal_places=2, blank=True, null=True)
    vencimento = models.DateField(blank=True, null=True)
    codigo = models.AutoField(primary_key=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'parcelas'


class ParcelasCanceladas(models.Model):
    parcela = models.OneToOneField(Parcelas, models.DO_NOTHING, db_column='parcela', primary_key=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)
    motivo = models.ForeignKey(Motivos, models.DO_NOTHING, db_column='motivo', blank=True, null=True)
    complemento = models.CharField(max_length=50, blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'parcelas_canceladas'


class ParcelasClientes(models.Model):
    parcela = models.OneToOneField(Parcelas, models.DO_NOTHING, db_column='parcela', primary_key=True)
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'parcelas_clientes'


class ParcelasDescontos(models.Model):
    parcela = models.OneToOneField('ParcelasQuitadas', models.DO_NOTHING, db_column='parcela', primary_key=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'parcelas_descontos'


class ParcelasDestino(models.Model):
    cd_parcela = models.ForeignKey(Parcelas, models.DO_NOTHING, db_column='cd_parcela', blank=True, null=True)
    parcela_quitada = models.ForeignKey('ParcelasQuitadas', models.DO_NOTHING, db_column='parcela_quitada', blank=True, null=True)
    id_caixa = models.IntegerField(blank=True, null=True)
    id_banco = models.IntegerField(blank=True, null=True)
    id_credito = models.IntegerField(blank=True, null=True)
    id_juros = models.IntegerField(blank=True, null=True)
    id_multa = models.IntegerField(blank=True, null=True)
    id_desconto = models.IntegerField(blank=True, null=True)
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'parcelas_destino'


class ParcelasDuplicata(models.Model):
    parcela = models.OneToOneField(Parcelas, models.DO_NOTHING, db_column='parcela', primary_key=True)
    duplicata = models.CharField(max_length=20)
    portador = models.ForeignKey(Bancos, models.DO_NOTHING, db_column='portador', blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)
    convenio = models.ForeignKey(ConvenioBancos, models.DO_NOTHING, db_column='convenio', blank=True, null=True)
    enviado_banco = models.BooleanField(blank=True, null=True)
    vencimento_gerado = models.DateField(blank=True, null=True)
    aceite = models.BooleanField(blank=True, null=True)
    datamovimentacao = models.DateTimeField(blank=True, null=True)
    documento = models.CharField(max_length=20)

    class Meta:
        managed = False
        db_table = 'parcelas_duplicata'
        unique_together = (('parcela', 'duplicata'),)


class ParcelasDuplicataExcluidas(models.Model):
    codigo = models.AutoField(primary_key=True)
    parcela = models.IntegerField(blank=True, null=True)
    duplicata = models.CharField(max_length=20, blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)
    ordem = models.IntegerField(blank=True, null=True)
    fatura = models.IntegerField(blank=True, null=True)
    vencimento = models.DateField(blank=True, null=True)
    convenio = models.IntegerField(blank=True, null=True)
    tipo = models.CharField(max_length=20, blank=True, null=True)
    vencimento_gerado = models.DateField(blank=True, null=True)
    aceite = models.BooleanField(blank=True, null=True)
    datamovimentacao = models.DateField(blank=True, null=True)
    documento = models.CharField(max_length=20)

    class Meta:
        managed = False
        db_table = 'parcelas_duplicata_excluidas'


class ParcelasDuplicatasLog(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    parcela = models.IntegerField()
    duplicata = models.CharField(max_length=20, blank=True, null=True)
    convenio = models.IntegerField()
    ordem = models.IntegerField(blank=True, null=True)
    fatura = models.IntegerField(blank=True, null=True)
    vencimento = models.DateField()
    lote = models.IntegerField(blank=True, null=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535)

    class Meta:
        managed = False
        db_table = 'parcelas_duplicatas_log'


class ParcelasEnvioBoleto(models.Model):
    codigo = models.AutoField(primary_key=True)
    parcela = models.ForeignKey(Parcelas, models.DO_NOTHING, db_column='parcela')
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    data = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'parcelas_envio_boleto'


class ParcelasEnvioLembrete(models.Model):
    codigo = models.AutoField(primary_key=True)
    parcela = models.ForeignKey(Parcelas, models.DO_NOTHING, db_column='parcela', blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    data = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'parcelas_envio_lembrete'


class ParcelasEspecie(models.Model):
    parcela = models.OneToOneField(Parcelas, models.DO_NOTHING, db_column='parcela', primary_key=True)
    especie = models.ForeignKey(Especies, models.DO_NOTHING, db_column='especie', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'parcelas_especie'


class ParcelasJuros(models.Model):
    parcela = models.OneToOneField('ParcelasQuitadas', models.DO_NOTHING, db_column='parcela', primary_key=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'parcelas_juros'


class ParcelasMarcadas(models.Model):
    parcela = models.OneToOneField(Parcelas, models.DO_NOTHING, db_column='parcela', primary_key=True)

    class Meta:
        managed = False
        db_table = 'parcelas_marcadas'


class ParcelasMultas(models.Model):
    parcela = models.OneToOneField('ParcelasQuitadas', models.DO_NOTHING, db_column='parcela', primary_key=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'parcelas_multas'


class ParcelasObservacoes(models.Model):
    parcela = models.OneToOneField(Parcelas, models.DO_NOTHING, db_column='parcela', primary_key=True)
    observacao = models.ForeignKey(ObservacoesMovimentos, models.DO_NOTHING, db_column='observacao')

    class Meta:
        managed = False
        db_table = 'parcelas_observacoes'


class ParcelasParciais(models.Model):
    parcela = models.OneToOneField(Parcelas, models.DO_NOTHING, db_column='parcela', primary_key=True)
    origem = models.ForeignKey('ParcelasQuitadas', models.DO_NOTHING, db_column='origem', blank=True, null=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535)

    class Meta:
        managed = False
        db_table = 'parcelas_parciais'


class ParcelasPontos(models.Model):
    parcela = models.OneToOneField(Parcelas, models.DO_NOTHING, db_column='parcela', primary_key=True)
    pontos = models.DecimalField(max_digits=65535, decimal_places=65535)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'parcelas_pontos'


class ParcelasPortador(models.Model):
    parcela = models.OneToOneField(Parcelas, models.DO_NOTHING, db_column='parcela', primary_key=True)
    portador = models.ForeignKey(Bancos, models.DO_NOTHING, db_column='portador', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'parcelas_portador'


class ParcelasQuitadas(models.Model):
    parcela = models.OneToOneField(Parcelas, models.DO_NOTHING, db_column='parcela', primary_key=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)
    valor_pago = models.DecimalField(max_digits=31, decimal_places=15, blank=True, null=True)
    data_pgto = models.DateField(blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'parcelas_quitadas'


class ParcelasQuitadasHistorico(models.Model):
    codigo = models.AutoField(primary_key=True)
    parcela = models.IntegerField(blank=True, null=True)
    ordem = models.ForeignKey(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', blank=True, null=True)
    valor_pago = models.DecimalField(max_digits=65535, decimal_places=65535)
    data_pgto = models.DateField()
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'parcelas_quitadas_historico'


class ParcelasTemp(models.Model):
    parcela = models.ForeignKey(Parcelas, models.DO_NOTHING, db_column='parcela', blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)
    pid = models.BigIntegerField()

    class Meta:
        managed = False
        db_table = 'parcelas_temp'


class ParcelasVencimentosAlterados(models.Model):
    parcela = models.ForeignKey(Parcelas, models.DO_NOTHING, db_column='parcela', blank=True, null=True)
    data = models.DateTimeField()
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    vencimento_atual = models.DateField(blank=True, null=True)
    vencimento_anterior = models.DateField(blank=True, null=True)
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'parcelas_vencimentos_alterados'


class Patrimonio(models.Model):
    npatrimonio = models.CharField(primary_key=True, max_length=20)
    descricao = models.CharField(max_length=50, blank=True, null=True)
    data_aqui = models.DateField(blank=True, null=True)
    valor = models.FloatField(blank=True, null=True)
    taxa_depre = models.FloatField(blank=True, null=True)
    quantidade = models.IntegerField(blank=True, null=True)
    centro_custo = models.ForeignKey(Centrocustos, models.DO_NOTHING, db_column='centro_custo', blank=True, null=True)
    fornecedor = models.ForeignKey(Fornecedores, models.DO_NOTHING, db_column='fornecedor', blank=True, null=True)
    nro_nota = models.IntegerField(blank=True, null=True)
    cod_index = models.ForeignKey(Indicesconversao, models.DO_NOTHING, db_column='cod_index', blank=True, null=True)
    departamento = models.ForeignKey(Departamentos, models.DO_NOTHING, db_column='departamento', blank=True, null=True)
    tempo_depre = models.IntegerField(blank=True, null=True)
    historico = models.ForeignKey(Historicos, models.DO_NOTHING, db_column='historico', blank=True, null=True)
    observacao = models.ForeignKey(Observacoes, models.DO_NOTHING, db_column='observacao', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'patrimonio'


class Pdv(models.Model):
    codigo = models.IntegerField(primary_key=True)
    comanda = models.IntegerField(blank=True, null=True)
    vendedor = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='vendedor', blank=True, null=True)
    produto = models.ForeignKey('Produtos', models.DO_NOTHING, db_column='produto', blank=True, null=True)
    descricao = models.CharField(max_length=60, blank=True, null=True)
    unidade = models.CharField(max_length=6, blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    preco_venda = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    preco_custo = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    desconto = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    total = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    obs1 = models.CharField(max_length=57, blank=True, null=True)
    obs2 = models.CharField(max_length=57, blank=True, null=True)
    obs3 = models.CharField(max_length=57, blank=True, null=True)
    cond_pgto = models.ForeignKey('Vencimentos', models.DO_NOTHING, db_column='cond_pgto', blank=True, null=True)
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'pdv'


class PdvImpressorasNaoFiscais(models.Model):
    codigo = models.AutoField(primary_key=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    datahora = models.DateTimeField()
    impressora = models.CharField(max_length=-1)
    changeid = models.CharField(max_length=-1)
    numero_hd = models.CharField(max_length=-1)
    datahora_envio = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'pdv_impressoras_nao_fiscais'


class PdvVendasAbertas(models.Model):
    numero_serie = models.CharField(primary_key=True, max_length=30)
    pedido = models.OneToOneField(CabecalhoOrdemServico, models.DO_NOTHING, db_column='pedido')

    class Meta:
        managed = False
        db_table = 'pdv_vendas_abertas'


class PdvVendasPendentesCancelamento(models.Model):
    ordem = models.OneToOneField(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem')
    coo = models.IntegerField(blank=True, null=True)
    serie = models.CharField(primary_key=True, max_length=-1)
    mensagem = models.CharField(max_length=-1, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'pdv_vendas_pendentes_cancelamento'


class PedidoCompra(models.Model):
    pedido = models.IntegerField(primary_key=True)
    data = models.DateField(blank=True, null=True)
    data_entrega = models.DateField(blank=True, null=True)
    fornecedor = models.ForeignKey(Fornecedores, models.DO_NOTHING, db_column='fornecedor', blank=True, null=True)
    status = models.CharField(max_length=1)
    data_fechamento = models.DateTimeField(blank=True, null=True)
    data_abertura = models.DateTimeField()
    data_cancelou = models.DateTimeField(blank=True, null=True)
    comprador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='comprador')
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    observacoes = models.ForeignKey(ObservacoesMovimentos, models.DO_NOTHING, db_column='observacoes', blank=True, null=True)
    mensagem = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'pedido_compra'


class PedidoCompraDataBase(models.Model):
    pedido = models.OneToOneField(PedidoCompra, models.DO_NOTHING, db_column='pedido', primary_key=True)
    data_base = models.DateField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'pedido_compra_data_base'


class PedidoCompraPagamento(models.Model):
    pedido = models.OneToOneField(PedidoCompra, models.DO_NOTHING, db_column='pedido', primary_key=True)
    condicao_pgto = models.ForeignKey('Vencimentos', models.DO_NOTHING, db_column='condicao_pgto', blank=True, null=True)
    portador = models.ForeignKey(Bancos, models.DO_NOTHING, db_column='portador', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'pedido_compra_pagamento'


class PedidoCompraParcela(models.Model):
    codigo = models.AutoField(primary_key=True)
    pedido = models.ForeignKey(PedidoCompra, models.DO_NOTHING, db_column='pedido', blank=True, null=True)
    parcela = models.IntegerField()
    vencimento = models.DateField()
    valor = models.DecimalField(max_digits=65535, decimal_places=65535)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    data_lancamento = models.DateField()
    hora_lancamento = models.TimeField()

    class Meta:
        managed = False
        db_table = 'pedido_compra_parcela'


class PedidoCompraTransportador(models.Model):
    pedido = models.OneToOneField(PedidoCompra, models.DO_NOTHING, db_column='pedido', primary_key=True)
    transportador = models.ForeignKey(Fornecedores, models.DO_NOTHING, db_column='transportador', blank=True, null=True)
    redespacho = models.ForeignKey(Fornecedores, models.DO_NOTHING, db_column='redespacho', blank=True, null=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    tipo_frete = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'pedido_compra_transportador'


class PedidoEmitirNota(models.Model):
    ordem = models.IntegerField()
    transportador = models.IntegerField()
    pid = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'pedido_emitir_nota'
        unique_together = (('ordem', 'pid'),)


class PedidoEmitirNotaAutomatica(models.Model):
    codigo = models.AutoField(primary_key=True)
    ordem = models.ForeignKey(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem')
    vendedor = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='vendedor')
    usersessionid = models.CharField(max_length=-1)

    class Meta:
        managed = False
        db_table = 'pedido_emitir_nota_automatica'


class PedidoEmitirNotaAutomaticaRetorno(models.Model):
    codigo = models.AutoField(primary_key=True)
    ordem = models.IntegerField()
    codigonota = models.IntegerField()
    retorno = models.IntegerField()
    mensagem = models.CharField(max_length=-1)

    class Meta:
        managed = False
        db_table = 'pedido_emitir_nota_automatica_retorno'


class PedidoSequencia(models.Model):
    ordem = models.OneToOneField(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', primary_key=True)
    imprimiu_pedido = models.BooleanField(blank=True, null=True)
    imprimiu_nota = models.BooleanField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'pedido_sequencia'


class PedidoSequenciaCompra(models.Model):
    nota = models.OneToOneField(CabecalhoNotaCompra, models.DO_NOTHING, db_column='nota', primary_key=True)

    class Meta:
        managed = False
        db_table = 'pedido_sequencia_compra'


class PedidosImportados(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_terminal = models.ForeignKey('TerminaisImportacao', models.DO_NOTHING, db_column='cd_terminal', blank=True, null=True)
    pedido_origem = models.IntegerField(blank=True, null=True)
    pedido_destino = models.IntegerField(blank=True, null=True)
    operador = models.IntegerField(blank=True, null=True)
    data = models.DateField()
    hora = models.TimeField()

    class Meta:
        managed = False
        db_table = 'pedidos_importados'


class PedidosImpressos(models.Model):
    codigo = models.AutoField(primary_key=True)
    ordem = models.ForeignKey(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', blank=True, null=True)
    reserva = models.ForeignKey(CabecalhoOrcamento, models.DO_NOTHING, db_column='reserva', blank=True, null=True)
    entrega_futura = models.ForeignKey(CabecalhoEntrega, models.DO_NOTHING, db_column='entrega_futura', blank=True, null=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    data = models.DateField()
    hora = models.TimeField()
    usuario_autorizou = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario_autorizou')

    class Meta:
        managed = False
        db_table = 'pedidos_impressos'


class PedidosProcedimentos(models.Model):
    cod_pedido = models.OneToOneField(CabecalhoOrdemServico, models.DO_NOTHING, db_column='cod_pedido', primary_key=True)
    cod_procedimento = models.ForeignKey('Procedimentos', models.DO_NOTHING, db_column='cod_procedimento', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'pedidos_procedimentos'


class PisCofins(models.Model):
    produto = models.OneToOneField('Produtos', models.DO_NOTHING, db_column='produto', primary_key=True)
    cst_pis = models.CharField(max_length=2, blank=True, null=True)
    pis_aliq = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    cst_cofins = models.CharField(max_length=2, blank=True, null=True)
    cofins_aliq = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    cst_pis_entrada = models.CharField(max_length=2, blank=True, null=True)
    pis_aliq_entrada = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    cst_cofins_entrada = models.CharField(max_length=2, blank=True, null=True)
    cofins_aliq_entrada = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    codigo_receita = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'pis_cofins'


class Placascep(models.Model):
    placa = models.CharField(primary_key=True, max_length=8)
    cep = models.ForeignKey(Cidades, models.DO_NOTHING, db_column='cep')
    codigo_antt = models.CharField(max_length=20)

    class Meta:
        managed = False
        db_table = 'placascep'


class PolaridadeMotor(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nome = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'polaridade_motor'


class Ponto(models.Model):
    funcionario = models.IntegerField(blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    horaent1 = models.TimeField(blank=True, null=True)
    horasai1 = models.TimeField(blank=True, null=True)
    horaent2 = models.TimeField(blank=True, null=True)
    horasai2 = models.TimeField(blank=True, null=True)
    tipo = models.CharField(max_length=2, blank=True, null=True)
    codigo = models.IntegerField(primary_key=True)
    entextra = models.TimeField(blank=True, null=True)
    saiextra = models.TimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'ponto'


class PontosClienteDesconto(models.Model):
    cliente = models.OneToOneField(Clientes, models.DO_NOTHING, db_column='cliente', primary_key=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'pontos_cliente_desconto'


class PontosClientesBloqueadosStandalone(models.Model):
    cliente = models.OneToOneField(Clientes, models.DO_NOTHING, db_column='cliente', primary_key=True)
    pontos = models.DecimalField(max_digits=65535, decimal_places=65535)

    class Meta:
        managed = False
        db_table = 'pontos_clientes_bloqueados_standalone'


class PontosClientesDisponiveisStandalone(models.Model):
    cliente = models.OneToOneField(Clientes, models.DO_NOTHING, db_column='cliente', primary_key=True)
    pontos = models.DecimalField(max_digits=65535, decimal_places=65535)

    class Meta:
        managed = False
        db_table = 'pontos_clientes_disponiveis_standalone'


class PontosControle(models.Model):
    codigo = models.AutoField(primary_key=True)
    parcela = models.ForeignKey(ParcelasQuitadas, models.DO_NOTHING, db_column='parcela', blank=True, null=True)
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente')
    pontos = models.DecimalField(max_digits=65535, decimal_places=65535)
    validade = models.IntegerField()
    data_operacao = models.DateField()
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    inclusao_manual = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'pontos_controle'


class PontosControleExtrato(models.Model):
    codigo = models.AutoField(primary_key=True)
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente')
    pontos_controle = models.ForeignKey(PontosControle, models.DO_NOTHING, db_column='pontos_controle')
    descricao = models.CharField(max_length=500)
    pontos = models.DecimalField(max_digits=65535, decimal_places=65535)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'pontos_controle_extrato'


class PontosControleExtratoReferencias(models.Model):
    codigo = models.AutoField(primary_key=True)
    extrato = models.ForeignKey(PontosControleExtrato, models.DO_NOTHING, db_column='extrato')
    pontos_controle = models.ForeignKey(PontosControle, models.DO_NOTHING, db_column='pontos_controle')
    pontos = models.DecimalField(max_digits=65535, decimal_places=65535)
    indexador = models.DecimalField(max_digits=65535, decimal_places=65535)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'pontos_controle_extrato_referencias'


class PontosControleHistoricoParcelasQuitadas(models.Model):
    codigo = models.AutoField(primary_key=True)
    pontos_controle = models.ForeignKey(PontosControle, models.DO_NOTHING, db_column='pontos_controle')
    parcela_historico = models.ForeignKey(ParcelasQuitadasHistorico, models.DO_NOTHING, db_column='parcela_historico', blank=True, null=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'pontos_controle_historico_parcelas_quitadas'


class PontosControleSaldo(models.Model):
    pontos_controle = models.OneToOneField(PontosControle, models.DO_NOTHING, db_column='pontos_controle', primary_key=True)
    saldo = models.DecimalField(max_digits=65535, decimal_places=65535)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'pontos_controle_saldo'


class PontosDescontosStandaloneConsumidos(models.Model):
    codigo = models.AutoField(primary_key=True)
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente')
    terminal = models.ForeignKey('TerminaisImportacao', models.DO_NOTHING, db_column='terminal')
    ordem_terminal = models.IntegerField()
    valor = models.DecimalField(max_digits=65535, decimal_places=65535)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'pontos_descontos_standalone_consumidos'


class PontosDevolucaoDiferenca(models.Model):
    codigo = models.AutoField(primary_key=True)
    devolucao = models.ForeignKey(CabecalhoDevolucao, models.DO_NOTHING, db_column='devolucao')
    pontos = models.DecimalField(max_digits=65535, decimal_places=65535)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'pontos_devolucao_diferenca'


class PontosParcelasDevolucaoHistorico(models.Model):
    codigo = models.AutoField(primary_key=True)
    parcela = models.ForeignKey(Parcelas, models.DO_NOTHING, db_column='parcela')
    devolucao = models.ForeignKey(CabecalhoDevolucao, models.DO_NOTHING, db_column='devolucao')
    pontos = models.DecimalField(max_digits=65535, decimal_places=65535)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'pontos_parcelas_devolucao_historico'


class PontosParcelasDevolucaoPendentes(models.Model):
    codigo = models.AutoField(primary_key=True)
    parcela = models.ForeignKey(Parcelas, models.DO_NOTHING, db_column='parcela')
    devolucao = models.ForeignKey(CabecalhoDevolucao, models.DO_NOTHING, db_column='devolucao')
    pontos = models.DecimalField(max_digits=65535, decimal_places=65535)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'pontos_parcelas_devolucao_pendentes'


class PosVenda(models.Model):
    cod_item = models.ForeignKey(ItensOrdemServico, models.DO_NOTHING, db_column='cod_item')
    data = models.DateField(blank=True, null=True)
    cd_observacao = models.ForeignKey(ObservacoesMovimentos, models.DO_NOTHING, db_column='cd_observacao', blank=True, null=True)
    vendedor = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='vendedor', blank=True, null=True)
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'pos_venda'


class PosVendaOs(models.Model):
    codigo = models.AutoField(primary_key=True)
    cod_item = models.ForeignKey(OsServicos, models.DO_NOTHING, db_column='cod_item')
    data = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    ndias = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'pos_venda_os'


class PotenciaMotor(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nome = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'potencia_motor'


class Procedimentos(models.Model):
    codigo = models.AutoField(primary_key=True)
    nome = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'procedimentos'


class Produtos(models.Model):
    codigo = models.IntegerField(primary_key=True)
    codigodefabrica = models.CharField(max_length=20, blank=True, null=True)
    descricao = models.CharField(max_length=50)
    unidade = models.ForeignKey('UnidadeMedidas', models.DO_NOTHING, db_column='unidade')
    ativo = models.DecimalField(max_digits=65535, decimal_places=65535)
    grupo = models.ForeignKey(GrupoProdutos, models.DO_NOTHING, db_column='grupo', blank=True, null=True)
    subgrupo = models.ForeignKey('SubgrupoProdutos', models.DO_NOTHING, db_column='subgrupo', blank=True, null=True)
    departamento = models.ForeignKey(Departamentos, models.DO_NOTHING, db_column='departamento', blank=True, null=True)
    estoqueminimo = models.DecimalField(max_digits=15, decimal_places=3)
    estoquemaximo = models.DecimalField(max_digits=15, decimal_places=3)
    quantidade = models.DecimalField(max_digits=20, decimal_places=3)
    embalagem = models.DecimalField(max_digits=15, decimal_places=3)
    precovenda = models.DecimalField(max_digits=65535, decimal_places=65535)
    precocusto = models.DecimalField(max_digits=65535, decimal_places=65535)
    precocontabil = models.DecimalField(max_digits=15, decimal_places=6)
    icms = models.DecimalField(max_digits=6, decimal_places=2)
    percdesconto = models.DecimalField(max_digits=6, decimal_places=2)
    vendersemestoque = models.DecimalField(max_digits=65535, decimal_places=65535)
    iss = models.DecimalField(max_digits=6, decimal_places=2)
    codbarra = models.CharField(max_length=14, blank=True, null=True)
    garantia = models.CharField(max_length=50, blank=True, null=True)
    baixa = models.CharField(max_length=1)
    classifica = models.ForeignKey(ClasseProdutos, models.DO_NOTHING, db_column='classifica', blank=True, null=True)
    tabela_preco = models.ForeignKey(CabecaPreco, models.DO_NOTHING, db_column='tabela_preco', blank=True, null=True)
    marca = models.CharField(max_length=1, blank=True, null=True)
    estoq_inicial = models.DecimalField(max_digits=15, decimal_places=2)
    class_fiscal = models.ForeignKey(ClassFiscal, models.DO_NOTHING, db_column='class_fiscal', blank=True, null=True)
    aplicacao = models.CharField(max_length=200, blank=True, null=True)
    cod_ncm = models.CharField(max_length=9, blank=True, null=True)
    localizacao = models.CharField(max_length=20)
    situacaotributaria = models.CharField(max_length=4)
    comissao = models.DecimalField(max_digits=65535, decimal_places=65535)
    observacoes = models.TextField(blank=True, null=True)
    preco_minimo = models.DecimalField(max_digits=65535, decimal_places=65535)
    carcaca = models.BooleanField()
    pesado = models.BooleanField()
    dimensao = models.CharField(max_length=10, blank=True, null=True)
    medidas = models.CharField(max_length=10, blank=True, null=True)
    pi = models.CharField(max_length=10, blank=True, null=True)
    classe = models.CharField(max_length=10, blank=True, null=True)
    peso_bruto = models.DecimalField(max_digits=65535, decimal_places=65535)
    peso_liquido = models.DecimalField(max_digits=65535, decimal_places=65535)
    montado = models.BooleanField(blank=True, null=True)
    ipi = models.DecimalField(max_digits=65535, decimal_places=65535)
    dias_contato = models.IntegerField(blank=True, null=True)
    funrural = models.DecimalField(max_digits=65535, decimal_places=65535)
    cfop_venda = models.IntegerField(blank=True, null=True)
    cfop_compra = models.IntegerField(blank=True, null=True)
    st_fora = models.CharField(max_length=4)
    importado = models.BooleanField(blank=True, null=True)
    tipo_etiqueta = models.IntegerField()
    dias_preco = models.IntegerField(blank=True, null=True)
    desc_catalogo = models.CharField(max_length=30, blank=True, null=True)
    tipo_mercadoria = models.CharField(max_length=2)
    genero_item = models.ForeignKey(GeneroItem, models.DO_NOTHING, db_column='genero_item', blank=True, null=True)
    conta_analitica = models.CharField(max_length=15, blank=True, null=True)
    envia_palm = models.BooleanField()
    grade = models.IntegerField(blank=True, null=True)
    md5 = models.CharField(max_length=-1)
    ipi_entrada = models.DecimalField(max_digits=65535, decimal_places=65535)
    codigo_cnae = models.ForeignKey(CnaeProdutos, models.DO_NOTHING, db_column='codigo_cnae', blank=True, null=True)
    credita_icmsst = models.BooleanField()
    codigo_anp = models.CharField(max_length=9, blank=True, null=True)
    recarga_celular = models.BooleanField()
    paranoprecopdv = models.BooleanField()
    cest = models.CharField(max_length=7, blank=True, null=True)
    unidade_tributavel = models.ForeignKey('UnidadeMedidas', models.DO_NOTHING, db_column='unidade_tributavel')
    pgnn = models.DecimalField(max_digits=10, decimal_places=4)
    pgni = models.DecimalField(max_digits=10, decimal_places=4)
    pglp = models.DecimalField(max_digits=10, decimal_places=4)
    ind_escala_relevante = models.IntegerField()
    beneficio = models.ForeignKey(Beneficio, models.DO_NOTHING, db_column='beneficio', blank=True, null=True)
    utiliza_lote = models.BooleanField()
    codigo_anvisa = models.CharField(max_length=13, blank=True, null=True)
    embalagem_venda = models.DecimalField(max_digits=15, decimal_places=3)
    iss_tributacao = models.ForeignKey('ProdutosIss', models.DO_NOTHING, db_column='iss_tributacao', blank=True, null=True)
    desonerado = models.ForeignKey(Desonerado, models.DO_NOTHING, db_column='desonerado', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'produtos'


class ProdutosAdicionais(models.Model):
    produto = models.OneToOneField(Produtos, models.DO_NOTHING, db_column='produto', primary_key=True)
    adicional = models.CharField(max_length=20, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'produtos_adicionais'


class ProdutosAliquotas(models.Model):
    codigo = models.AutoField(primary_key=True)
    situacao_tributaria = models.CharField(max_length=3, blank=True, null=True)
    nome = models.CharField(max_length=50, blank=True, null=True)
    aliquota_icms = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'produtos_aliquotas'


class ProdutosAlteraQuantidade(models.Model):
    codigo = models.AutoField(primary_key=True)
    produto = models.ForeignKey(Produtos, models.DO_NOTHING, db_column='produto', blank=True, null=True)
    data = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    qtd_anterior = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    qtd_nova = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    md5 = models.CharField(max_length=-1)
    chave = models.IntegerField()
    modulo_sistema = models.CharField(max_length=-1)

    class Meta:
        managed = False
        db_table = 'produtos_altera_quantidade'


class ProdutosAlteracoesSped(models.Model):
    codigo = models.AutoField(primary_key=True)
    data_inicial = models.DateField(blank=True, null=True)
    data_final = models.DateField(blank=True, null=True)
    produto = models.ForeignKey(Produtos, models.DO_NOTHING, db_column='produto', blank=True, null=True)
    descricao = models.CharField(max_length=50, blank=True, null=True)
    competencia = models.CharField(max_length=7, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'produtos_alteracoes_sped'


class ProdutosCadastroImportacaoXml(models.Model):
    codigo = models.AutoField(primary_key=True)
    produto = models.ForeignKey(Produtos, models.DO_NOTHING, db_column='produto')
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'produtos_cadastro_importacao_xml'


class ProdutosCodbarra(models.Model):
    codigo = models.AutoField(primary_key=True)
    codbarra = models.CharField(max_length=14)
    padraonf = models.BooleanField()
    produto = models.ForeignKey(Produtos, models.DO_NOTHING, db_column='produto')
    origem = models.IntegerField()
    data = models.DateTimeField()
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')

    class Meta:
        managed = False
        db_table = 'produtos_codbarra'


class ProdutosComposicao(models.Model):
    produto = models.ForeignKey(Produtos, models.DO_NOTHING, db_column='produto')
    produto_item = models.ForeignKey(Produtos, models.DO_NOTHING, db_column='produto_item')
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    codigo = models.AutoField(primary_key=True)
    lanca_composicao_pdv = models.BooleanField()
    quantidade_fixa = models.BooleanField()
    tipo_produto = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'produtos_composicao'
        unique_together = (('produto', 'produto_item'),)


class ProdutosEmprestados(models.Model):
    codigo = models.AutoField(primary_key=True)
    data = models.DateField()
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente', blank=True, null=True)
    produto = models.ForeignKey(Produtos, models.DO_NOTHING, db_column='produto', blank=True, null=True)
    previsao = models.DateField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    data_lancamento = models.DateField()
    hora_lancamento = models.TimeField()
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    custo = models.DecimalField(max_digits=65535, decimal_places=65535)
    custocontabil = models.DecimalField(max_digits=65535, decimal_places=65535)
    preco_venda = models.DecimalField(max_digits=65535, decimal_places=65535)
    descricao = models.CharField(max_length=100, blank=True, null=True)
    codbarra = models.CharField(max_length=14)

    class Meta:
        managed = False
        db_table = 'produtos_emprestados'


class ProdutosEmprestadosDevolvidos(models.Model):
    codigo = models.AutoField(primary_key=True)
    produto = models.ForeignKey(ProdutosEmprestados, models.DO_NOTHING, db_column='produto', blank=True, null=True)
    data = models.DateField()
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    data_lancamento = models.DateField()
    hora_lancamento = models.TimeField()

    class Meta:
        managed = False
        db_table = 'produtos_emprestados_devolvidos'


class ProdutosEmprestei(models.Model):
    codigo = models.AutoField(primary_key=True)
    data = models.DateField()
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente', blank=True, null=True)
    produto = models.ForeignKey(Produtos, models.DO_NOTHING, db_column='produto', blank=True, null=True)
    previsao = models.DateField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    data_lancamento = models.DateField()
    hora_lancamento = models.TimeField()
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535)
    custo = models.DecimalField(max_digits=65535, decimal_places=65535)
    custocontabil = models.DecimalField(max_digits=65535, decimal_places=65535)
    preco_venda = models.DecimalField(max_digits=65535, decimal_places=65535)
    descricao = models.CharField(max_length=100, blank=True, null=True)
    codbarra = models.CharField(max_length=14)

    class Meta:
        managed = False
        db_table = 'produtos_emprestei'


class ProdutosEmpresteiDevolvi(models.Model):
    codigo = models.AutoField(primary_key=True)
    produto = models.ForeignKey(ProdutosEmprestei, models.DO_NOTHING, db_column='produto', blank=True, null=True)
    data = models.DateField()
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    data_lancamento = models.DateField()
    hora_lancamento = models.TimeField()

    class Meta:
        managed = False
        db_table = 'produtos_emprestei_devolvi'


class ProdutosExcluidos(models.Model):
    codigo = models.IntegerField(blank=True, null=True)
    descricao = models.CharField(max_length=-1, blank=True, null=True)
    unidade = models.CharField(max_length=6, blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'produtos_excluidos'


class ProdutosImagens(models.Model):
    produto = models.ForeignKey(Produtos, models.DO_NOTHING, db_column='produto', blank=True, null=True)
    imagem = models.CharField(max_length=256, blank=True, null=True)
    departamento = models.ForeignKey(Departamentos, models.DO_NOTHING, db_column='departamento', blank=True, null=True)
    tipo = models.CharField(max_length=1, blank=True, null=True)
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'produtos_imagens'


class ProdutosIpi(models.Model):
    cod_produto = models.OneToOneField(Produtos, models.DO_NOTHING, db_column='cod_produto', primary_key=True)
    cst = models.CharField(max_length=2, blank=True, null=True)
    enquadramento = models.CharField(max_length=3, blank=True, null=True)
    cst_entrada = models.CharField(max_length=2, blank=True, null=True)
    enquadramento_entrada = models.CharField(max_length=3, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'produtos_ipi'


class ProdutosIss(models.Model):
    codigo = models.AutoField(primary_key=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    datahora = models.DateTimeField()
    descricao = models.CharField(max_length=50)
    tributa = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'produtos_iss'


class ProdutosNcmIbpt(models.Model):
    ncm = models.CharField(max_length=-1, blank=True, null=True)
    ex = models.CharField(max_length=-1, blank=True, null=True)
    codigo = models.AutoField(primary_key=True)
    tipo = models.CharField(max_length=-1, blank=True, null=True)
    descricao = models.CharField(max_length=-1, blank=True, null=True)
    nacionalfederal = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    importadosfederal = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    estadual = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    municipal = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    vigenciainicio = models.DateField(blank=True, null=True)
    vigenciafim = models.DateField(blank=True, null=True)
    chave = models.CharField(max_length=-1, blank=True, null=True)
    versao = models.CharField(max_length=-1, blank=True, null=True)
    fonte = models.CharField(max_length=-1, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'produtos_ncm_ibpt'


class ProdutosNegativados(models.Model):
    produto = models.ForeignKey(Produtos, models.DO_NOTHING, db_column='produto', blank=True, null=True)
    saldo_estoque = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    custo = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    custo_contabil = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    data = models.DateField()
    hora = models.TimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'produtos_negativados'


class ProdutosNutricionais(models.Model):
    codigo = models.OneToOneField(Produtos, models.DO_NOTHING, db_column='codigo', primary_key=True)
    informacoes_nutricionais = models.ForeignKey(InformacoesNutricionais, models.DO_NOTHING, db_column='informacoes_nutricionais', blank=True, null=True)
    informacoes_extras = models.ForeignKey(InformacoesExtras, models.DO_NOTHING, db_column='informacoes_extras', blank=True, null=True)
    taras = models.ForeignKey('Taras', models.DO_NOTHING, db_column='taras', blank=True, null=True)
    fracionadores = models.ForeignKey(Fracionadores, models.DO_NOTHING, db_column='fracionadores', blank=True, null=True)
    conservacoes = models.ForeignKey(Conservacoes, models.DO_NOTHING, db_column='conservacoes', blank=True, null=True)
    extra1 = models.ForeignKey(Extra1, models.DO_NOTHING, db_column='extra1', blank=True, null=True)
    extra2 = models.ForeignKey(Extra2, models.DO_NOTHING, db_column='extra2', blank=True, null=True)
    imprime_data = models.BooleanField(blank=True, null=True)
    imprime_validade = models.BooleanField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'produtos_nutricionais'


class ProdutosPendentesTransmitir(models.Model):
    codigo = models.AutoField(primary_key=True)
    competencia_ano_mes = models.CharField(max_length=6)
    data_operacao = models.DateTimeField()
    enviou_sefaz = models.BooleanField(blank=True, null=True)
    recibo = models.CharField(max_length=100)
    dh_envio = models.DateTimeField(blank=True, null=True)
    mensagem_retorno = models.CharField(max_length=1000, blank=True, null=True)
    retorno = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'produtos_pendentes_transmitir'


class ProdutosPontos(models.Model):
    produto = models.OneToOneField(Produtos, models.DO_NOTHING, db_column='produto', primary_key=True)
    indexador = models.DecimalField(max_digits=65535, decimal_places=65535)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'produtos_pontos'


class ProdutosPontosLog(models.Model):
    codigo = models.AutoField(primary_key=True)
    produto = models.ForeignKey(Produtos, models.DO_NOTHING, db_column='produto')
    indexador_antigo = models.DecimalField(max_digits=65535, decimal_places=65535)
    indexador_novo = models.DecimalField(max_digits=65535, decimal_places=65535)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'produtos_pontos_log'


class ProdutosReceitas(models.Model):
    produto = models.OneToOneField(Produtos, models.DO_NOTHING, db_column='produto', primary_key=True)
    observacao = models.ForeignKey(Observacoes, models.DO_NOTHING, db_column='observacao', blank=True, null=True)
    dose = models.CharField(max_length=50, blank=True, null=True)
    class_toxicologica = models.CharField(max_length=50, blank=True, null=True)
    classificacao = models.CharField(max_length=50, blank=True, null=True)
    inst_seguranca = models.TextField(blank=True, null=True)
    diagnostico = models.CharField(max_length=255, blank=True, null=True)
    apresentacao = models.TextField(blank=True, null=True)
    carencia = models.TextField(blank=True, null=True)
    incompativel = models.TextField(blank=True, null=True)
    principioativo = models.CharField(max_length=255, blank=True, null=True)
    grupo_quimico = models.CharField(max_length=255, blank=True, null=True)
    formulacao = models.CharField(max_length=50, blank=True, null=True)
    concentracao = models.CharField(max_length=50, blank=True, null=True)
    fitotoxidade = models.CharField(max_length=50, blank=True, null=True)
    equipamento = models.CharField(max_length=50, blank=True, null=True)
    aplicacao = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'produtos_receitas'


class ProdutosRelacaoYandeh(models.Model):
    codigo = models.AutoField(primary_key=True)
    produto = models.ForeignKey(Produtos, models.DO_NOTHING, db_column='produto')
    id_yandeh = models.CharField(max_length=20, blank=True, null=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador', blank=True, null=True)
    data = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'produtos_relacao_yandeh'


class ProdutosSimilares(models.Model):
    codigo = models.AutoField(primary_key=True)
    produto = models.ForeignKey(Produtos, models.DO_NOTHING, db_column='produto', blank=True, null=True)
    produto_similar = models.ForeignKey(Produtos, models.DO_NOTHING, db_column='produto_similar', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'produtos_similares'


class ProdutosSubstTabelas(models.Model):
    produto = models.OneToOneField(Produtos, models.DO_NOTHING, db_column='produto', primary_key=True)
    tabela = models.ForeignKey('SubstTabelas', models.DO_NOTHING, db_column='tabela', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'produtos_subst_tabelas'


class ProdutosTemp(models.Model):
    codigo = models.IntegerField(blank=True, null=True)
    codigodefabrica = models.CharField(max_length=20, blank=True, null=True)
    descricao = models.CharField(max_length=50, blank=True, null=True)
    unidade = models.CharField(max_length=2, blank=True, null=True)
    grupo = models.IntegerField(blank=True, null=True)
    departamento = models.IntegerField(blank=True, null=True)
    precovenda = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    icms = models.DecimalField(max_digits=6, decimal_places=2, blank=True, null=True)
    situacaotributaria = models.CharField(max_length=4, blank=True, null=True)
    vendersemestoque = models.DecimalField(max_digits=1, decimal_places=0, blank=True, null=True)
    iss = models.DecimalField(max_digits=6, decimal_places=2, blank=True, null=True)
    codbarra = models.CharField(max_length=14, blank=True, null=True)
    grupodescricao = models.CharField(max_length=50, blank=True, null=True)
    depardescricao = models.CharField(max_length=50, blank=True, null=True)
    baixa = models.CharField(max_length=1, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'produtos_temp'


class ProdutosUltimaCompra(models.Model):
    produto = models.OneToOneField(Produtos, models.DO_NOTHING, db_column='produto', primary_key=True)
    item_compra = models.ForeignKey(ItensNotaCompra, models.DO_NOTHING, db_column='item_compra')

    class Meta:
        managed = False
        db_table = 'produtos_ultima_compra'


class Profissoes(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nome = models.CharField(max_length=50, blank=True, null=True)
    codigo_financeira = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'profissoes'


class PromissoriasEmitidas(models.Model):
    codigo = models.AutoField(primary_key=True)
    parcela = models.ForeignKey(Parcelas, models.DO_NOTHING, db_column='parcela', blank=True, null=True)
    data = models.DateField()
    hora = models.TimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'promissorias_emitidas'


class Promocao(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    descricao = models.CharField(max_length=60)
    validade_inicial = models.DateField()
    validade_final = models.DateField()
    tipo_promocao = models.IntegerField()
    quantidade_promocao = models.DecimalField(max_digits=15, decimal_places=3)
    quantidade_brinde = models.DecimalField(max_digits=15, decimal_places=3)
    tipo_desconto = models.IntegerField()
    desconto = models.DecimalField(max_digits=15, decimal_places=3)

    class Meta:
        managed = False
        db_table = 'promocao'


class ProvisaoGuiasDifaFcp(models.Model):
    codigo = models.AutoField(primary_key=True)
    competencia = models.CharField(max_length=7)
    vencimento = models.DateField()
    valor = models.DecimalField(max_digits=15, decimal_places=2)
    codigo_obrigacao = models.CharField(max_length=3)
    codigo_receita = models.CharField(max_length=30)
    estado = models.ForeignKey(Estados, models.DO_NOTHING, db_column='estado')
    data_operacao = models.DateTimeField()
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    tipooperacao = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'provisao_guias_difa_fcp'


class RateioContasApagar(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_apagar = models.ForeignKey(Apagar, models.DO_NOTHING, db_column='cd_apagar')
    cd_centrocusto = models.ForeignKey(Centrocustos, models.DO_NOTHING, db_column='cd_centrocusto')
    total = models.DecimalField(max_digits=15, decimal_places=2)
    cd_operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='cd_operador')
    data = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'rateio_contas_apagar'


class RateioContasApagarNotaCompra(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_apagar = models.ForeignKey(PagamentoNotaCompra, models.DO_NOTHING, db_column='cd_apagar')
    cd_centrocusto = models.ForeignKey(Centrocustos, models.DO_NOTHING, db_column='cd_centrocusto')
    total = models.DecimalField(max_digits=15, decimal_places=2)
    cd_operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='cd_operador')
    data = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'rateio_contas_apagar_nota_compra'


class Rateiocontas(models.Model):
    codigo = models.IntegerField(primary_key=True)
    planoconta = models.ForeignKey(Historicos, models.DO_NOTHING, db_column='planoconta')
    centrocusto = models.ForeignKey(Centrocustos, models.DO_NOTHING, db_column='centrocusto')
    percentual = models.DecimalField(max_digits=6, decimal_places=2)

    class Meta:
        managed = False
        db_table = 'rateiocontas'


class Receitas(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    status = models.CharField(max_length=1)
    datahorafechamentocancelamento = models.DateTimeField(blank=True, null=True)
    usuariofechamentocancelamento = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuariofechamentocancelamento', blank=True, null=True)
    tecnicoagricola = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='tecnicoagricola')
    datahoraemissao = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'receitas'


class ReceitasItens(models.Model):
    codigo = models.AutoField(primary_key=True)
    item_venda = models.ForeignKey(ItensOrdemServico, models.DO_NOTHING, db_column='item_venda', blank=True, null=True)
    item_reserva = models.ForeignKey(ItensOrcamento, models.DO_NOTHING, db_column='item_reserva', blank=True, null=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'receitas_itens'


class ReceitasItensDetalhes(models.Model):
    codigo = models.AutoField(primary_key=True)
    item_receita = models.ForeignKey(ReceitasItens, models.DO_NOTHING, db_column='item_receita', blank=True, null=True)
    area = models.CharField(max_length=50)
    adquirir = models.CharField(max_length=50)
    utilizar = models.CharField(max_length=-1)
    aplicacoes = models.CharField(max_length=50)
    observacoes = models.ForeignKey(Observacoes, models.DO_NOTHING, db_column='observacoes', blank=True, null=True)
    dose = models.CharField(max_length=50)
    cultura = models.CharField(max_length=-1)
    diagnostico = models.CharField(max_length=-1)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    equipamento = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'receitas_itens_detalhes'


class ReceitasItensReferencias(models.Model):
    codigo = models.AutoField(primary_key=True)
    receita = models.ForeignKey(Receitas, models.DO_NOTHING, db_column='receita')
    item_receita_detalhe = models.ForeignKey(ReceitasItensDetalhes, models.DO_NOTHING, db_column='item_receita_detalhe', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'receitas_itens_referencias'


class RecibosAvulsos(models.Model):
    recibo = models.OneToOneField('RecibosImpressos', models.DO_NOTHING, db_column='recibo', primary_key=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'recibos_avulsos'


class RecibosImpressos(models.Model):
    codigo = models.AutoField(primary_key=True)
    data = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'recibos_impressos'


class RecibosParcelasApagar(models.Model):
    recibo = models.ForeignKey(RecibosImpressos, models.DO_NOTHING, db_column='recibo', blank=True, null=True)
    parcela = models.ForeignKey(Apagar, models.DO_NOTHING, db_column='parcela', blank=True, null=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'recibos_parcelas_apagar'


class RecibosParcelasReceber(models.Model):
    recibo = models.ForeignKey(RecibosImpressos, models.DO_NOTHING, db_column='recibo', blank=True, null=True)
    parcela = models.ForeignKey(Parcelas, models.DO_NOTHING, db_column='parcela', blank=True, null=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'recibos_parcelas_receber'


class ReducoeszImpressas(models.Model):
    numero_serie = models.CharField(max_length=-1)
    data_movimento = models.DateField()
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'reducoesz_impressas'


class ReferenciaCaixa(models.Model):
    codigo = models.AutoField(primary_key=True)
    caixa_origem = models.IntegerField(blank=True, null=True)
    caixa_destino = models.IntegerField(blank=True, null=True)
    terminal = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'referencia_caixa'


class ReferenciasNumeroSerie(models.Model):
    codigo = models.AutoField(primary_key=True)
    mapa_60m = models.ForeignKey(MapaEcf60M, models.DO_NOTHING, db_column='mapa_60m', blank=True, null=True)
    doc_gerenciais = models.ForeignKey(DocumentosGerenciais, models.DO_NOTHING, db_column='doc_gerenciais', blank=True, null=True)
    mapa_pagamento = models.ForeignKey(MapaEcfFormasPagamento, models.DO_NOTHING, db_column='mapa_pagamento', blank=True, null=True)
    numero_serie = models.CharField(max_length=-1)
    exportou = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'referencias_numero_serie'


class Regioes(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nome = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'regioes'


class Registro(models.Model):
    nome = models.CharField(max_length=60, blank=True, null=True)
    razao = models.CharField(max_length=60, blank=True, null=True)
    endereco = models.CharField(max_length=50, blank=True, null=True)
    bairro = models.CharField(max_length=50, blank=True, null=True)
    cep = models.CharField(max_length=9, blank=True, null=True)
    cidade = models.CharField(max_length=50, blank=True, null=True)
    uf = models.CharField(max_length=2, blank=True, null=True)
    fone = models.CharField(max_length=14, blank=True, null=True)
    cnpj = models.CharField(primary_key=True, max_length=18)
    ie = models.CharField(max_length=18, blank=True, null=True)
    url = models.CharField(max_length=50, blank=True, null=True)
    email = models.CharField(max_length=50, blank=True, null=True)
    operacional = models.DecimalField(max_digits=10, decimal_places=3, blank=True, null=True)
    imp_federais = models.DecimalField(max_digits=10, decimal_places=3, blank=True, null=True)
    nboleto = models.IntegerField(blank=True, null=True)
    fiscal = models.BooleanField()
    ultimo_acesso = models.DateTimeField(blank=True, null=True)
    dvua = models.IntegerField(blank=True, null=True)
    chave = models.IntegerField(blank=True, null=True)
    datareg = models.DateField(blank=True, null=True)
    simples = models.IntegerField()
    industria = models.IntegerField(blank=True, null=True)
    conexoes = models.IntegerField()
    inscricao_municipal = models.CharField(max_length=15, blank=True, null=True)
    cnae = models.CharField(max_length=7, blank=True, null=True)
    contato = models.CharField(max_length=50, blank=True, null=True)
    prestador_servico = models.BooleanField(blank=True, null=True)
    inscricao_substituto = models.CharField(max_length=50, blank=True, null=True)
    inscricao_sulframa = models.CharField(max_length=50, blank=True, null=True)
    apuracaoregimecaixa = models.BooleanField()
    cooperativa = models.BooleanField(blank=True, null=True)
    checagem = models.CharField(max_length=32, blank=True, null=True)
    mei = models.BooleanField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'registro'


class RegistroTemporario(models.Model):
    nome = models.CharField(max_length=60, blank=True, null=True)
    razao = models.CharField(max_length=60, blank=True, null=True)
    endereco = models.CharField(max_length=50, blank=True, null=True)
    bairro = models.CharField(max_length=50, blank=True, null=True)
    cep = models.CharField(max_length=9, blank=True, null=True)
    cidade = models.CharField(max_length=50, blank=True, null=True)
    uf = models.CharField(max_length=2, blank=True, null=True)
    fone = models.CharField(max_length=14, blank=True, null=True)
    cnpj = models.CharField(max_length=18, blank=True, null=True)
    ie = models.CharField(max_length=18, blank=True, null=True)
    url = models.CharField(max_length=50, blank=True, null=True)
    email = models.CharField(max_length=50, blank=True, null=True)
    operacional = models.DecimalField(max_digits=10, decimal_places=3, blank=True, null=True)
    imp_federais = models.DecimalField(max_digits=10, decimal_places=3, blank=True, null=True)
    nboleto = models.IntegerField(blank=True, null=True)
    fiscal = models.BooleanField(blank=True, null=True)
    ultimo_acesso = models.DateTimeField(blank=True, null=True)
    dvua = models.IntegerField(blank=True, null=True)
    chave = models.IntegerField(blank=True, null=True)
    datareg = models.DateField(blank=True, null=True)
    simples = models.IntegerField(blank=True, null=True)
    industria = models.IntegerField(blank=True, null=True)
    conexoes = models.IntegerField(blank=True, null=True)
    inscricao_municipal = models.CharField(max_length=15, blank=True, null=True)
    cnae = models.CharField(max_length=7, blank=True, null=True)
    contato = models.CharField(max_length=50, blank=True, null=True)
    prestador_servico = models.BooleanField(blank=True, null=True)
    inscricao_substituto = models.CharField(max_length=50, blank=True, null=True)
    inscricao_sulframa = models.CharField(max_length=50, blank=True, null=True)
    apuracaoregimecaixa = models.BooleanField(blank=True, null=True)
    cooperativa = models.BooleanField(blank=True, null=True)
    checagem = models.CharField(max_length=32, blank=True, null=True)
    mei = models.BooleanField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'registro_temporario'


class RelatorioRequisitado(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.IntegerField()
    user_session_id = models.CharField(max_length=-1)
    sistema = models.IntegerField()
    relatorio = models.CharField(max_length=-1)
    parametros = models.CharField(max_length=-1)
    processando = models.BooleanField()
    pdf = models.BinaryField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'relatorio_requisitado'


class Relatorios(models.Model):
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    linha = models.CharField(max_length=255, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'relatorios'


class Remessa(models.Model):
    numero = models.AutoField(primary_key=True)
    data = models.DateField(blank=True, null=True)
    fornecedor = models.ForeignKey(Fornecedores, models.DO_NOTHING, db_column='fornecedor', blank=True, null=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador', blank=True, null=True)
    vendedor = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='vendedor', blank=True, null=True)
    status = models.CharField(max_length=1, blank=True, null=True)
    transportadora = models.ForeignKey(Fornecedores, models.DO_NOTHING, db_column='transportadora', blank=True, null=True)
    frete = models.DecimalField(max_digits=12, decimal_places=3)
    observacao = models.ForeignKey(ObservacoesMovimentos, models.DO_NOTHING, db_column='observacao', blank=True, null=True)
    tipo = models.CharField(max_length=1)
    outras = models.DecimalField(max_digits=12, decimal_places=3)
    seguro = models.DecimalField(max_digits=12, decimal_places=3)
    acrescimo = models.DecimalField(max_digits=12, decimal_places=3)
    desconto = models.DecimalField(max_digits=12, decimal_places=3)
    tipo_frete = models.IntegerField()
    icms_frete = models.DecimalField(max_digits=12, decimal_places=3)
    data_cancelamento = models.DateTimeField(blank=True, null=True)
    usuario_cancelamento = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario_cancelamento', blank=True, null=True)
    finalidade = models.IntegerField()
    data_hora_fechamento = models.DateTimeField(blank=True, null=True)
    usuario_fechamento = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario_fechamento', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'remessa'


class RemessaEntrada(models.Model):
    numero = models.AutoField(primary_key=True)
    data = models.DateField(blank=True, null=True)
    fornecedor = models.ForeignKey(Fornecedores, models.DO_NOTHING, db_column='fornecedor', blank=True, null=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador', blank=True, null=True)
    status = models.CharField(max_length=1, blank=True, null=True)
    transportadora = models.ForeignKey(Fornecedores, models.DO_NOTHING, db_column='transportadora', blank=True, null=True)
    frete = models.DecimalField(max_digits=12, decimal_places=3, blank=True, null=True)
    observacao = models.ForeignKey(ObservacoesMovimentos, models.DO_NOTHING, db_column='observacao', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'remessa_entrada'


class Remuneracoes(models.Model):
    codigo = models.IntegerField(primary_key=True)
    salario = models.DecimalField(max_digits=15, decimal_places=0, blank=True, null=True)
    observacoes = models.CharField(max_length=200, blank=True, null=True)
    insalubre = models.DecimalField(max_digits=15, decimal_places=0, blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    vendedor = models.IntegerField(blank=True, null=True)
    ferias = models.BooleanField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'remuneracoes'


class Rendimento(models.Model):
    codigo = models.AutoField(primary_key=True)
    codigo_produto = models.ForeignKey(ItensProducao, models.DO_NOTHING, db_column='codigo_produto', blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535)
    tipo_operacao = models.CharField(max_length=1)
    quantidade_adicional = models.DecimalField(max_digits=65535, decimal_places=65535)

    class Meta:
        managed = False
        db_table = 'rendimento'


class Representantes(models.Model):
    fornecedor = models.OneToOneField(Fornecedores, models.DO_NOTHING, db_column='fornecedor', primary_key=True)
    nome = models.CharField(max_length=50, blank=True, null=True)
    endereco = models.CharField(max_length=50, blank=True, null=True)
    bairro = models.CharField(max_length=30, blank=True, null=True)
    cep = models.ForeignKey(Cidades, models.DO_NOTHING, db_column='cep', blank=True, null=True)
    cxpostal = models.CharField(max_length=15, blank=True, null=True)
    fone = models.CharField(max_length=15, blank=True, null=True)
    fax = models.CharField(max_length=15, blank=True, null=True)
    celular = models.CharField(max_length=15, blank=True, null=True)
    email = models.CharField(max_length=50, blank=True, null=True)
    cpf = models.CharField(max_length=18, blank=True, null=True)
    rg = models.CharField(max_length=20, blank=True, null=True)
    datanasc = models.DateField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'representantes'


class ReservaCliente(models.Model):
    codigo = models.AutoField(primary_key=True)
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente', blank=True, null=True)
    produto = models.ForeignKey(Produtos, models.DO_NOTHING, db_column='produto', blank=True, null=True)
    descricao = models.CharField(max_length=60, blank=True, null=True)
    quantidade = models.DecimalField(max_digits=10, decimal_places=3, blank=True, null=True)
    data = models.DateField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'reserva_cliente'


class ReservaEntrega(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_reserva = models.ForeignKey(CabecalhoOrcamento, models.DO_NOTHING, db_column='cd_reserva', blank=True, null=True)
    hora_entrega = models.TimeField(blank=True, null=True)
    local_entrega = models.CharField(max_length=-1, blank=True, null=True)
    cod_loja = models.ForeignKey(Filiais, models.DO_NOTHING, db_column='cod_loja', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'reserva_entrega'


class ResponsavelTecnico(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    cnpj = models.CharField(max_length=-1)
    contato = models.CharField(max_length=60)
    email = models.CharField(max_length=60)
    telefone = models.CharField(max_length=-1)

    class Meta:
        managed = False
        db_table = 'responsavel_tecnico'


class Rotas(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nome = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'rotas'


class Sat(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    numero_hd = models.CharField(max_length=8)
    numero_serie = models.CharField(max_length=30)
    modelo = models.CharField(max_length=30)
    modelo_str = models.CharField(max_length=30)
    codigo_ativacao = models.CharField(max_length=32)
    assinatura = models.CharField(max_length=500)
    layout = models.DecimalField(max_digits=65535, decimal_places=65535)
    chave_validador = models.CharField(max_length=500)
    codigo_estabelecimento_tef = models.CharField(max_length=500)
    ativo = models.BooleanField()
    motivo_desativacao = models.CharField(max_length=100)
    usuario_desativacao = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario_desativacao', blank=True, null=True)
    datahora_desativacao = models.DateTimeField(blank=True, null=True)
    porta_com = models.CharField(max_length=5)

    class Meta:
        managed = False
        db_table = 'sat'


class SatComando(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    pid = models.BigIntegerField()
    terminal = models.ForeignKey(Sat, models.DO_NOTHING, db_column='terminal')
    pdv = models.ForeignKey('Terminais', models.DO_NOTHING, db_column='pdv')
    comando = models.IntegerField()
    processando = models.BooleanField()
    ordem = models.ForeignKey(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', blank=True, null=True)
    mensagem = models.CharField(max_length=-1)
    motivo_cancelamento = models.ForeignKey(Motivos, models.DO_NOTHING, db_column='motivo_cancelamento', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'sat_comando'


class SatComandoIntermediario(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    pid = models.IntegerField()
    terminal = models.ForeignKey(Sat, models.DO_NOTHING, db_column='terminal')
    pdv = models.ForeignKey('Terminais', models.DO_NOTHING, db_column='pdv')
    comando = models.IntegerField()
    mensagem = models.CharField(max_length=-1)

    class Meta:
        managed = False
        db_table = 'sat_comando_intermediario'


class SatComandoRetorno(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    pid = models.BigIntegerField()
    terminal = models.ForeignKey(Sat, models.DO_NOTHING, db_column='terminal')
    pdv = models.ForeignKey('Terminais', models.DO_NOTHING, db_column='pdv')
    comando = models.IntegerField()
    codigo_retorno = models.IntegerField()
    mensagem = models.CharField(max_length=-1)

    class Meta:
        managed = False
        db_table = 'sat_comando_retorno'


class SatComandoRetornoIntermediario(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    pid = models.IntegerField()
    terminal = models.ForeignKey(Sat, models.DO_NOTHING, db_column='terminal')
    pdv = models.ForeignKey('Terminais', models.DO_NOTHING, db_column='pdv')
    comando = models.IntegerField()
    mensagem = models.CharField(max_length=-1)

    class Meta:
        managed = False
        db_table = 'sat_comando_retorno_intermediario'


class SequenciaNotas(models.Model):
    codigo = models.AutoField(primary_key=True)
    serie = models.CharField(max_length=-1)
    nota = models.IntegerField()
    tipo_emissao = models.CharField(max_length=-1)
    modelo = models.CharField(max_length=-1)
    ambiente = models.IntegerField(blank=True, null=True)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    datahora = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'sequencia_notas'
        unique_together = (('serie', 'modelo', 'tipo_emissao', 'ambiente'),)


class SequenciaNotasLogAjusteNumeracao(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    usuario_autorizou = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario_autorizou')
    serie = models.CharField(max_length=-1)
    nota_antiga = models.IntegerField()
    nota_nova = models.IntegerField()
    tipo_emissao = models.CharField(max_length=-1)
    modelo = models.CharField(max_length=-1)
    ambiente = models.IntegerField(blank=True, null=True)
    motivo = models.CharField(max_length=-1)

    class Meta:
        managed = False
        db_table = 'sequencia_notas_log_ajuste_numeracao'


class Seriais(models.Model):
    serial = models.CharField(primary_key=True, max_length=30)
    produto = models.ForeignKey(Produtos, models.DO_NOTHING, db_column='produto', blank=True, null=True)
    disponivel = models.CharField(max_length=3, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'seriais'


class Servicos(models.Model):
    codigo = models.AutoField(primary_key=True)
    responsavel = models.CharField(max_length=30, blank=True, null=True)
    entrega = models.DateField(blank=True, null=True)
    status = models.CharField(max_length=1, blank=True, null=True)
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente', blank=True, null=True)
    abertura = models.DateField(blank=True, null=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador', blank=True, null=True)
    usuario_cancelamento = models.IntegerField(blank=True, null=True)
    hora_entrega = models.TimeField(blank=True, null=True)
    hora_abertura = models.TimeField(blank=True, null=True)
    obs = models.ForeignKey(ObservacoesMovimentos, models.DO_NOTHING, db_column='obs', blank=True, null=True)
    servicos = models.ForeignKey(ObservacoesMovimentos, models.DO_NOTHING, db_column='servicos', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'servicos'


class ServicosIntervalo(models.Model):
    codigo = models.AutoField(primary_key=True)
    cod_item = models.ForeignKey(OsServicos, models.DO_NOTHING, db_column='cod_item', blank=True, null=True)
    cod_usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='cod_usuario', blank=True, null=True)
    inicio = models.DateTimeField()
    fim = models.DateTimeField(blank=True, null=True)
    status = models.CharField(max_length=20, blank=True, null=True)
    tipo = models.ForeignKey('TipoHorasOs', models.DO_NOTHING, db_column='tipo', blank=True, null=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'servicos_intervalo'


class ServicosProcedimentos(models.Model):
    cod_ordem = models.OneToOneField(Servicos, models.DO_NOTHING, db_column='cod_ordem', primary_key=True)
    cod_procedimento = models.ForeignKey(Procedimentos, models.DO_NOTHING, db_column='cod_procedimento', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'servicos_procedimentos'


class ServidorStandalone(models.Model):
    codigo = models.AutoField(primary_key=True)
    ip = models.CharField(max_length=-1)
    porta = models.IntegerField()
    base = models.CharField(max_length=-1)
    usuario_banco = models.CharField(max_length=-1)
    senha = models.CharField(max_length=-1)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    terminal = models.IntegerField()
    md5_terminal = models.CharField(max_length=32)

    class Meta:
        managed = False
        db_table = 'servidor_standalone'


class SitTrib(models.Model):
    codigo = models.CharField(primary_key=True, max_length=3)
    descricao = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'sit_trib'


class Spc(models.Model):
    codigo = models.AutoField(primary_key=True)
    data = models.DateField(blank=True, null=True)
    resultado = models.CharField(max_length=500, blank=True, null=True)
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente', blank=True, null=True)
    sistema_protecao = models.ForeignKey(Fornecedores, models.DO_NOTHING, db_column='sistema_protecao', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'spc'


class Standalonelogerrosimportacao(models.Model):
    codigo = models.AutoField(primary_key=True)
    terminal = models.ForeignKey('TerminaisImportacao', models.DO_NOTHING, db_column='terminal')
    erro = models.CharField(max_length=-1)
    datahora = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'standalonelogerrosimportacao'


class Standalonelogexportacaoterminais(models.Model):
    codigo = models.AutoField(primary_key=True)
    terminal = models.ForeignKey('TerminaisImportacao', models.DO_NOTHING, db_column='terminal')
    datahoraexportacao = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'standalonelogexportacaoterminais'


class Standalonelogimportacao(models.Model):
    codigo = models.AutoField(primary_key=True)
    tabela = models.ForeignKey('Standalonetabelasimportacao', models.DO_NOTHING, db_column='tabela')
    campo = models.CharField(max_length=-1)
    valorantigo = models.CharField(max_length=-1)
    valornovo = models.CharField(max_length=-1)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    terminal = models.ForeignKey('TerminaisImportacao', models.DO_NOTHING, db_column='terminal')

    class Meta:
        managed = False
        db_table = 'standalonelogimportacao'


class Standalonelogregistros(models.Model):
    codigo = models.AutoField(primary_key=True)
    tabela = models.ForeignKey('Standalonetabelasimportacao', models.DO_NOTHING, db_column='tabela')
    campo = models.CharField(max_length=-1)
    valor = models.CharField(max_length=-1)
    apagado = models.BooleanField()
    datahora = models.DateTimeField()
    oid_registros = models.BigIntegerField()

    class Meta:
        managed = False
        db_table = 'standalonelogregistros'


class Standalonetabelasimportacao(models.Model):
    codigo = models.AutoField(primary_key=True)
    tabela = models.CharField(max_length=-1)

    class Meta:
        managed = False
        db_table = 'standalonetabelasimportacao'


class SubgrupoProdutos(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nome = models.CharField(max_length=70, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'subgrupo_produtos'


class SubstAliquotas(models.Model):
    codigo = models.AutoField(primary_key=True)
    tabela = models.ForeignKey('SubstTabelas', models.DO_NOTHING, db_column='tabela', blank=True, null=True)
    uf = models.CharField(max_length=2, blank=True, null=True)
    aliquota = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    cst = models.CharField(max_length=4, blank=True, null=True)
    reducao_mva = models.DecimalField(max_digits=65535, decimal_places=65535)
    mensagem = models.TextField(blank=True, null=True)
    aliquota_interna_mercadoria = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    icms_reducao = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    icms_aliquota = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    calcula_percentual_reducao = models.BooleanField()
    reducao_base_icms_st = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    autoriza_descontar_icms = models.BooleanField()
    utiliza_diferenca_interestadual = models.BooleanField()
    mva_consumo_proprio = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    autoriza_desc_icms_usoeconsumo = models.BooleanField()
    utiliza_diferencial_aliquota = models.BooleanField(blank=True, null=True)
    reducao_diferencial_aliquota = models.DecimalField(max_digits=20, decimal_places=2)
    carga_efetiva_aliquota_interestadual = models.BooleanField()
    carga_efetiva_aliquota_interna_destinatario = models.BooleanField()
    fundo_combate_pobreza = models.DecimalField(max_digits=20, decimal_places=2)
    calculo_antigo = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'subst_aliquotas'


class SubstTabelas(models.Model):
    codigo = models.AutoField(primary_key=True)
    descricao = models.CharField(max_length=50, blank=True, null=True)
    lei = models.CharField(max_length=1000, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'subst_tabelas'


class Taras(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nome = models.CharField(max_length=20, blank=True, null=True)
    tara = models.DecimalField(max_digits=65535, decimal_places=65535)

    class Meta:
        managed = False
        db_table = 'taras'


class TaxasOperadoras(models.Model):
    codigo = models.AutoField(primary_key=True)
    operadora = models.ForeignKey(Fornecedores, models.DO_NOTHING, db_column='operadora')
    parcelas = models.IntegerField()
    taxa = models.DecimalField(max_digits=65535, decimal_places=65535)
    intervalo = models.IntegerField()
    descricao = models.CharField(max_length=500)
    complemento = models.CharField(max_length=500)
    datahora = models.DateTimeField()
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')
    operacao = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'taxas_operadoras'
        unique_together = (('operadora', 'parcelas', 'operacao'),)


class TefEmitirNota(models.Model):
    codigo = models.BigAutoField(primary_key=True)
    cnpj = models.CharField(max_length=-1)
    bandeira = models.CharField(max_length=-1)
    credito = models.BooleanField()
    debito = models.BooleanField()
    nsu = models.CharField(max_length=50)
    pid = models.BigIntegerField()
    dados = models.TextField()

    class Meta:
        managed = False
        db_table = 'tef_emitir_nota'


class TempPedidosPalm(models.Model):
    pedido = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'temp_pedidos_palm'


class TensaoMotor(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nome = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'tensao_motor'


class Terminais(models.Model):
    codigo = models.IntegerField(primary_key=True)
    numero_serie = models.CharField(max_length=30, blank=True, null=True)
    numero_hd = models.CharField(max_length=8)
    impressora = models.CharField(max_length=30, blank=True, null=True)
    porta_com = models.CharField(max_length=5, blank=True, null=True)
    balanca = models.CharField(max_length=30, blank=True, null=True)
    porta_com_balanca = models.CharField(max_length=5, blank=True, null=True)
    modelo_ecf = models.CharField(max_length=20, blank=True, null=True)
    codigo_ecf = models.IntegerField(blank=True, null=True)
    arredonda = models.CharField(max_length=1)
    tipo_ecf = models.CharField(max_length=7, blank=True, null=True)
    marca_ecf = models.CharField(max_length=20, blank=True, null=True)
    firmware = models.CharField(max_length=10, blank=True, null=True)
    modelo_str = models.CharField(max_length=20, blank=True, null=True)
    reducoes_restantes = models.IntegerField()
    tipo_paf_ecf = models.IntegerField()
    velocidade_com = models.IntegerField()
    datasw = models.DateField(blank=True, null=True)
    horasw = models.TimeField(blank=True, null=True)
    mfadicional = models.BooleanField()
    ativo = models.BooleanField()
    motivodesativacao = models.CharField(max_length=50, blank=True, null=True)
    acionaguilhotina = models.IntegerField()
    modelo_sat = models.CharField(max_length=30, blank=True, null=True)
    codigo_ativacao_sat = models.CharField(max_length=32, blank=True, null=True)
    md5_cupom = models.CharField(max_length=32)
    modelo_impressao_sat = models.IntegerField()
    assinatura_sat = models.CharField(max_length=500)
    gerencial_padrao = models.IntegerField()
    gerencial_identificacao_paf = models.IntegerField()
    gerencial_parametros_configuracao = models.IntegerField()
    utilizatef = models.BooleanField()
    fornecedor_tef = models.ForeignKey(Fornecedores, models.DO_NOTHING, db_column='fornecedor_tef', blank=True, null=True)
    dataenviotef = models.DateField(blank=True, null=True)
    guilhotina = models.BooleanField()
    guilhotina_parcial = models.BooleanField()
    paridade = models.IntegerField()
    stop_bits = models.IntegerField()
    data_bits = models.IntegerField()
    velocidade_com_balanca = models.IntegerField()
    handshake = models.IntegerField()
    layout_sat = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    certificado = models.CharField(max_length=50)
    codigo_impressao_caixa = models.IntegerField()
    chave_validador_sat = models.CharField(max_length=-1)
    codigo_estabelecimento_tef = models.CharField(max_length=-1)
    tipo_impressao = models.IntegerField()
    servidor_sat = models.ForeignKey(Sat, models.DO_NOTHING, db_column='servidor_sat', blank=True, null=True)
    modelo_impressao = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'terminais'


class TerminaisAliquotas(models.Model):
    codigo = models.AutoField(primary_key=True)
    terminal = models.ForeignKey(Terminais, models.DO_NOTHING, db_column='terminal', blank=True, null=True)
    numero_serie = models.CharField(max_length=-1, blank=True, null=True)
    aliquota = models.CharField(max_length=2, blank=True, null=True)
    icms = models.DecimalField(max_digits=65535, decimal_places=65535)
    tipo = models.CharField(max_length=1, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'terminais_aliquotas'


class TerminaisDados(models.Model):
    codigo = models.AutoField(primary_key=True)
    terminal = models.ForeignKey(Terminais, models.DO_NOTHING, db_column='terminal', blank=True, null=True)
    numero_serie = models.CharField(max_length=-1)
    cnpj_usuario = models.CharField(max_length=-1, blank=True, null=True)
    cnpj_desenvolvedora = models.CharField(max_length=-1, blank=True, null=True)
    nome_comercial = models.CharField(max_length=-1, blank=True, null=True)
    md5_executavel = models.CharField(max_length=-1, blank=True, null=True)
    md5 = models.CharField(max_length=-1, blank=True, null=True)
    exportou = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'terminais_dados'


class TerminaisFormasPgto(models.Model):
    codigo = models.AutoField(primary_key=True)
    terminal = models.ForeignKey(Terminais, models.DO_NOTHING, db_column='terminal', blank=True, null=True)
    numero_serie = models.CharField(max_length=30, blank=True, null=True)
    codigo_formapgto = models.CharField(max_length=2, blank=True, null=True)
    descricao = models.CharField(max_length=50, blank=True, null=True)
    especie = models.ForeignKey(Especies, models.DO_NOTHING, db_column='especie', blank=True, null=True)
    tecla = models.CharField(max_length=3)

    class Meta:
        managed = False
        db_table = 'terminais_formas_pgto'


class TerminaisGt(models.Model):
    numero_serie = models.CharField(primary_key=True, max_length=-1)
    gt = models.DecimalField(max_digits=65535, decimal_places=65535)

    class Meta:
        managed = False
        db_table = 'terminais_gt'


class TerminaisImportacao(models.Model):
    codigo = models.AutoField(primary_key=True)
    descricao = models.CharField(max_length=50, blank=True, null=True)
    base = models.CharField(max_length=50, blank=True, null=True)
    ip = models.CharField(max_length=15, blank=True, null=True)
    porta = models.IntegerField(blank=True, null=True)
    online = models.BooleanField(blank=True, null=True)
    datahora_ultima_importacao = models.DateTimeField(blank=True, null=True)
    ativo = models.BooleanField()
    md5_terminal = models.CharField(max_length=32)

    class Meta:
        managed = False
        db_table = 'terminais_importacao'


class TerminaisImportacaoErros(models.Model):
    codigo = models.AutoField(primary_key=True)
    terminal = models.ForeignKey(TerminaisImportacao, models.DO_NOTHING, db_column='terminal', blank=True, null=True)
    erro = models.CharField(max_length=-1, blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'terminais_importacao_erros'


class TerminaisLogados(models.Model):
    serie = models.ForeignKey('TerminaisSeries', models.DO_NOTHING, db_column='serie')
    nome_computador = models.CharField(max_length=30)
    ip = models.CharField(max_length=40, blank=True, null=True)
    numero_hd = models.CharField(max_length=100)
    tipo_sistema = models.CharField(max_length=1)
    data_conexao = models.DateTimeField()
    pid = models.BigIntegerField()
    ativo = models.BooleanField()
    chave = models.CharField(max_length=50)
    enviado = models.BooleanField()
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'terminais_logados'


class TerminaisMensagens(models.Model):
    numero_hd = models.CharField(max_length=8)
    data_operacao = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'terminais_mensagens'


class TerminaisSeries(models.Model):
    serie = models.CharField(primary_key=True, max_length=13)
    chave = models.CharField(max_length=50)
    data_operacao = models.DateTimeField()
    cnpj = models.ForeignKey(Registro, models.DO_NOTHING, db_column='cnpj')
    checagem = models.DateTimeField()
    bloqueado = models.BooleanField()
    master = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'terminais_series'


class TerminalMovel(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    nome = models.CharField(max_length=60)
    numero_serie = models.CharField(max_length=60)
    ativo = models.BooleanField()
    motivo_desativacao = models.CharField(max_length=100)
    usuario_desativacao = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario_desativacao', blank=True, null=True)
    datahora_desativacao = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'terminal_movel'


class TerminalMovelNfce(models.Model):
    codigo = models.AutoField(primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')
    terminal = models.ForeignKey(TerminalMovel, models.DO_NOTHING, db_column='terminal')
    modelo = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'terminal_movel_nfce'


class TerminalMovelSistema(models.Model):
    codigo = models.AutoField(primary_key=True)
    terminalmovel = models.ForeignKey(TerminalMovel, models.DO_NOTHING, db_column='terminalmovel')
    data_operacao = models.DateTimeField()
    operador_registro = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador_registro', blank=True, null=True)
    sistema = models.IntegerField(blank=True, null=True)
    vendedor = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='vendedor')
    ativo = models.BooleanField()
    motivo_desativacao = models.CharField(max_length=100)
    usuario_desativacao = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario_desativacao', blank=True, null=True)
    datahora_desativacao = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'terminal_movel_sistema'


class TipoHorasOs(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nome = models.CharField(max_length=50, blank=True, null=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'tipo_horas_os'


class TipoMercadoriaInventario(models.Model):
    indice = models.IntegerField(blank=True, null=True)
    tipo = models.CharField(primary_key=True, max_length=2)
    descricao = models.CharField(max_length=30, blank=True, null=True)
    inventario = models.BooleanField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'tipo_mercadoria_inventario'


class TmpBoletos(models.Model):
    local_pgto = models.CharField(max_length=150, blank=True, null=True)
    plano = models.IntegerField(blank=True, null=True)
    parcela = models.IntegerField(blank=True, null=True)
    n_document = models.CharField(max_length=20, blank=True, null=True)
    vencimento = models.DateField(blank=True, null=True)
    dt_process = models.DateField(blank=True, null=True)
    valor_doc = models.DecimalField(max_digits=12, decimal_places=2, blank=True, null=True)
    cedente = models.CharField(max_length=50, blank=True, null=True)
    cod_cedente = models.DecimalField(max_digits=30, decimal_places=0, blank=True, null=True)
    dv_cedente = models.CharField(max_length=2, blank=True, null=True)
    dt_document = models.DateField(blank=True, null=True)
    data_emissao = models.DateField(blank=True, null=True)
    nosso_nume = models.CharField(max_length=20, blank=True, null=True)
    aceite = models.CharField(max_length=1, blank=True, null=True)
    dv_n_numer = models.CharField(max_length=3)
    carteira = models.IntegerField(blank=True, null=True)
    banco = models.IntegerField(blank=True, null=True)
    dv_banco = models.IntegerField(blank=True, null=True)
    cod_moeda = models.IntegerField(blank=True, null=True)
    convenio = models.IntegerField(blank=True, null=True)
    agencia = models.CharField(max_length=10, blank=True, null=True)
    dv_agencia = models.CharField(max_length=5, blank=True, null=True)
    variacao = models.IntegerField(blank=True, null=True)
    posto = models.IntegerField(blank=True, null=True)
    instruc_1 = models.CharField(max_length=120, blank=True, null=True)
    instruc_2 = models.CharField(max_length=120, blank=True, null=True)
    instruc_3 = models.CharField(max_length=120, blank=True, null=True)
    instruc_4 = models.CharField(max_length=120, blank=True, null=True)
    instruc_5 = models.CharField(max_length=120, blank=True, null=True)
    instruc_6 = models.CharField(max_length=120, blank=True, null=True)
    sacado_1 = models.CharField(max_length=150, blank=True, null=True)
    sacado_2 = models.CharField(max_length=150, blank=True, null=True)
    sacado_3 = models.CharField(max_length=150, blank=True, null=True)
    sigcb = models.BooleanField(blank=True, null=True)
    linha_digitavel = models.CharField(max_length=54, blank=True, null=True)
    codigo_barras = models.CharField(max_length=47, blank=True, null=True)
    cod_cedente_str = models.CharField(max_length=30, blank=True, null=True)
    nosso_numero_str = models.CharField(max_length=30, blank=True, null=True)
    usuario = models.IntegerField()
    carteira_sigla = models.CharField(max_length=10)
    avalista_1 = models.CharField(max_length=150, blank=True, null=True)
    cnpj_cedente = models.CharField(max_length=18, blank=True, null=True)
    endereco_cedente = models.CharField(max_length=120, blank=True, null=True)
    convenio_unicred = models.BooleanField(blank=True, null=True)
    cep_cedente = models.CharField(max_length=9, blank=True, null=True)
    cidade_cedente = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'tmp_boletos'


class TmpFafNfCancelada(models.Model):
    data = models.DateField(blank=True, null=True)
    total = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    desconto = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    acrescimo = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    total_liquido = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    nome = models.CharField(max_length=50, blank=True, null=True)
    cpf_cnpj = models.CharField(max_length=18, blank=True, null=True)
    nota = models.IntegerField(blank=True, null=True)
    nota_serie = models.CharField(max_length=3, blank=True, null=True)
    nota_chave_nfe = models.CharField(max_length=-1, blank=True, null=True)
    nota_modelo = models.CharField(max_length=4, blank=True, null=True)
    codigo = models.IntegerField(blank=True, null=True)
    operacao = models.TextField(blank=True, null=True)
    cliente = models.IntegerField(blank=True, null=True)
    cabecalho_ordem_servico = models.IntegerField(blank=True, null=True)
    ordem_producao = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'tmp_faf_nf_cancelada'


class TmpNotaPdv(models.Model):
    usuario = models.OneToOneField('Vendedores', models.DO_NOTHING, db_column='usuario', primary_key=True)
    data_operacao = models.DateTimeField()
    nota = models.IntegerField()
    modelo = models.CharField(max_length=2)
    serie = models.IntegerField()
    data_emissao = models.DateField()
    data_saida = models.DateField()

    class Meta:
        managed = False
        db_table = 'tmp_nota_pdv'


class TmpRelPdv(models.Model):
    codigo = models.AutoField(primary_key=True)
    registro = models.CharField(max_length=-1, blank=True, null=True)
    linha = models.CharField(max_length=-1, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'tmp_rel_pdv'


class TransacoesTef(models.Model):
    codigo = models.AutoField(primary_key=True)
    ordem = models.ForeignKey(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem', blank=True, null=True)
    especie = models.IntegerField(blank=True, null=True)
    nsu = models.CharField(max_length=20, blank=True, null=True)
    coo = models.IntegerField(blank=True, null=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    dados = models.TextField(blank=True, null=True)
    rede = models.CharField(max_length=20, blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)
    operador = models.IntegerField()
    credito = models.BooleanField()
    finalizacao = models.CharField(max_length=50)
    debito = models.BooleanField()
    usagoc = models.BooleanField()

    class Meta:
        managed = False
        db_table = 'transacoes_tef'


class TransacoesTefAprovadas(models.Model):
    transacao = models.OneToOneField(TransacoesTef, models.DO_NOTHING, db_column='transacao', primary_key=True)
    data_aprovacao = models.DateField(blank=True, null=True)
    hora_aprovacao = models.TimeField(blank=True, null=True)
    usuario_aprovacao = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'transacoes_tef_aprovadas'


class TransacoesTefCanceladas(models.Model):
    transacao = models.OneToOneField(TransacoesTef, models.DO_NOTHING, db_column='transacao', primary_key=True)
    data_cancelamento = models.DateField(blank=True, null=True)
    hora_cancelamento = models.TimeField(blank=True, null=True)
    usuario_cancelamento = models.IntegerField()
    dados = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'transacoes_tef_canceladas'


class TransacoesTefControle(models.Model):
    codigo = models.AutoField(primary_key=True)
    valor = models.DecimalField(max_digits=65535, decimal_places=65535)
    taxa = models.DecimalField(max_digits=65535, decimal_places=65535)
    valor_liquido = models.DecimalField(max_digits=65535, decimal_places=65535)
    previsao = models.DateField()
    operadora = models.ForeignKey(Fornecedores, models.DO_NOTHING, db_column='operadora')
    cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cliente')
    num_parcela = models.IntegerField()
    identificacao = models.CharField(max_length=20)
    operacao = models.IntegerField()
    datahora = models.DateTimeField()
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')

    class Meta:
        managed = False
        db_table = 'transacoes_tef_controle'


class TransacoesTefControleCanceladas(models.Model):
    transacao = models.OneToOneField(TransacoesTefControle, models.DO_NOTHING, db_column='transacao', primary_key=True)
    motivo = models.ForeignKey(Motivos, models.DO_NOTHING, db_column='motivo')
    complemento = models.CharField(max_length=50)
    datahora = models.DateTimeField()
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')

    class Meta:
        managed = False
        db_table = 'transacoes_tef_controle_canceladas'


class TransacoesTefControleEstornadas(models.Model):
    codigo = models.AutoField(primary_key=True)
    transacao = models.ForeignKey(TransacoesTefControle, models.DO_NOTHING, db_column='transacao')
    valor = models.DecimalField(max_digits=16, decimal_places=2)
    taxa_antecipacao = models.DecimalField(max_digits=7, decimal_places=2)
    datahora = models.DateTimeField()
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')

    class Meta:
        managed = False
        db_table = 'transacoes_tef_controle_estornadas'


class TransacoesTefControleMarcadas(models.Model):
    transacao = models.OneToOneField(TransacoesTefControle, models.DO_NOTHING, db_column='transacao', primary_key=True)

    class Meta:
        managed = False
        db_table = 'transacoes_tef_controle_marcadas'


class TransacoesTefControleParcelas(models.Model):
    codigo = models.AutoField(primary_key=True)
    transacao = models.ForeignKey(TransacoesTefControle, models.DO_NOTHING, db_column='transacao')
    parcela = models.ForeignKey(Parcelas, models.DO_NOTHING, db_column='parcela')
    datahora = models.DateTimeField()
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')

    class Meta:
        managed = False
        db_table = 'transacoes_tef_controle_parcelas'


class TransacoesTefControleQuitadas(models.Model):
    transacao = models.OneToOneField(TransacoesTefControle, models.DO_NOTHING, db_column='transacao', primary_key=True)
    valor = models.DecimalField(max_digits=16, decimal_places=2)
    taxa_antecipacao = models.DecimalField(max_digits=7, decimal_places=2)
    data = models.DateField()
    conta = models.ForeignKey(Ctabancarias, models.DO_NOTHING, db_column='conta')
    datahora = models.DateTimeField()
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador')

    class Meta:
        managed = False
        db_table = 'transacoes_tef_controle_quitadas'


class TransacoesTefImpressas(models.Model):
    transacao = models.OneToOneField(TransacoesTef, models.DO_NOTHING, db_column='transacao', primary_key=True)

    class Meta:
        managed = False
        db_table = 'transacoes_tef_impressas'


class TransacoesTefParcelas(models.Model):
    transacao = models.OneToOneField(TransacoesTef, models.DO_NOTHING, db_column='transacao', primary_key=True)
    parcela = models.ForeignKey(Parcelas, models.DO_NOTHING, db_column='parcela')

    class Meta:
        managed = False
        db_table = 'transacoes_tef_parcelas'
        unique_together = (('transacao', 'parcela'),)


class TransacoesTemp(models.Model):
    transacao = models.IntegerField(blank=True, null=True)
    dados = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'transacoes_temp'


class TransacoesTempAprovadas(models.Model):
    transacao = models.IntegerField(blank=True, null=True)
    dados = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'transacoes_temp_aprovadas'


class Transferencia(models.Model):
    codpedido = models.AutoField(primary_key=True)
    data = models.DateField(blank=True, null=True)
    codfilial = models.ForeignKey(Filiais, models.DO_NOTHING, db_column='codfilial', blank=True, null=True)
    codoperador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='codoperador', blank=True, null=True)
    separador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='separador', blank=True, null=True)
    confirmado = models.CharField(max_length=3, blank=True, null=True)
    cod_cliente = models.ForeignKey(Clientes, models.DO_NOTHING, db_column='cod_cliente', blank=True, null=True)
    desc_cliente = models.CharField(max_length=74, blank=True, null=True)
    carregada = models.CharField(max_length=3, blank=True, null=True)
    criada_por = models.CharField(max_length=-1, blank=True, null=True)
    data_confirmado = models.DateField(blank=True, null=True)
    hora_confirmado = models.TimeField(blank=True, null=True)
    data_cancelamento = models.DateTimeField(blank=True, null=True)
    usuario_cancelamento = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario_cancelamento', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'transferencia'


class TransferenciaCompraFilial(models.Model):
    pedido = models.OneToOneField(Transferencia, models.DO_NOTHING, db_column='pedido', primary_key=True)
    datahora = models.DateTimeField()
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario')

    class Meta:
        managed = False
        db_table = 'transferencia_compra_filial'


class TransferenciaEntrada(models.Model):
    codpedido = models.AutoField(primary_key=True)
    data = models.DateField(blank=True, null=True)
    codfilial = models.ForeignKey(Filiais, models.DO_NOTHING, db_column='codfilial')
    codoperador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='codoperador', blank=True, null=True)
    separador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='separador', blank=True, null=True)
    confirmado = models.CharField(max_length=3, blank=True, null=True)
    desc_cliente = models.CharField(max_length=74, blank=True, null=True)
    data_confirmado = models.DateField(blank=True, null=True)
    hora_confirmado = models.TimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'transferencia_entrada'
        unique_together = (('codpedido', 'codfilial'),)


class Transmitidos(models.Model):
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador', blank=True, null=True)
    data = models.DateField(blank=True, null=True)
    hora = models.TimeField(blank=True, null=True)
    origem = models.CharField(max_length=18, blank=True, null=True)
    tipo = models.CharField(max_length=50, blank=True, null=True)
    inseridos = models.IntegerField(blank=True, null=True)
    alterados = models.IntegerField(blank=True, null=True)
    codigo = models.AutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'transmitidos'


class UnidadeMedidas(models.Model):
    unidade = models.CharField(primary_key=True, max_length=6)
    descricao = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'unidade_medidas'


class UsuariosLogados(models.Model):
    pid = models.DecimalField(primary_key=True, max_digits=65535, decimal_places=65535)
    usuario = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='usuario', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'usuarios_logados'


class ValidandoNotas(models.Model):
    pid = models.BigIntegerField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'validando_notas'


class ValoresDefaultJanelaDinamica(models.Model):
    codigo = models.AutoField(primary_key=True)
    modulo_sistema = models.ForeignKey(ModuloSistema, models.DO_NOTHING, db_column='modulo_sistema')
    field = models.CharField(max_length=-1, blank=True, null=True)
    value = models.CharField(max_length=-1, blank=True, null=True)
    data_hora = models.DateTimeField(blank=True, null=True)
    usuario = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'valores_default_janela_dinamica'


class Veiculos(models.Model):
    placa = models.CharField(primary_key=True, max_length=8)
    tipo = models.CharField(max_length=50, blank=True, null=True)
    cor = models.CharField(max_length=15, blank=True, null=True)
    marca = models.CharField(max_length=15, blank=True, null=True)
    modelo = models.CharField(max_length=15, blank=True, null=True)
    km = models.DecimalField(max_digits=8, decimal_places=0, blank=True, null=True)
    chassi = models.CharField(max_length=20, blank=True, null=True)
    ano = models.DecimalField(max_digits=4, decimal_places=0, blank=True, null=True)
    serie = models.CharField(max_length=15, blank=True, null=True)
    combustivel = models.CharField(max_length=10, blank=True, null=True)
    observacao1 = models.CharField(max_length=50, blank=True, null=True)
    observacao2 = models.CharField(max_length=50, blank=True, null=True)
    observacao3 = models.CharField(max_length=50, blank=True, null=True)
    observacao4 = models.CharField(max_length=50, blank=True, null=True)
    renavam = models.CharField(max_length=15, blank=True, null=True)
    exercicio = models.CharField(max_length=10, blank=True, null=True)
    cidade_emplaca = models.CharField(max_length=30, blank=True, null=True)
    valor = models.DecimalField(max_digits=15, decimal_places=3, blank=True, null=True)
    parc_financiada = models.IntegerField(blank=True, null=True)
    cod_produto = models.ForeignKey(Produtos, models.DO_NOTHING, db_column='cod_produto', blank=True, null=True)
    carga = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'veiculos'


class VelocidadeInternet(models.Model):
    codigo = models.AutoField(primary_key=True)
    nome = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'velocidade_internet'


class Vencimentos(models.Model):
    codigo = models.IntegerField(primary_key=True)
    parcelas = models.IntegerField(blank=True, null=True)
    descricao = models.CharField(max_length=500, blank=True, null=True)
    mensalista = models.CharField(max_length=1, blank=True, null=True)
    juros = models.DecimalField(max_digits=20, decimal_places=7, blank=True, null=True)
    desconto = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    complemento = models.CharField(max_length=500, blank=True, null=True)
    ativo = models.BooleanField(blank=True, null=True)
    parcela_mesmo_mes = models.BooleanField(blank=True, null=True)
    visualizar_na_venda = models.BooleanField()
    exporta_palm = models.BooleanField(blank=True, null=True)
    frete = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    comissao = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    valor_minimo = models.DecimalField(max_digits=65535, decimal_places=65535)
    portador = models.ForeignKey(Bancos, models.DO_NOTHING, db_column='portador', blank=True, null=True)
    especie = models.ForeignKey(Especies, models.DO_NOTHING, db_column='especie', blank=True, null=True)
    taxafinanceira = models.DecimalField(max_digits=65535, decimal_places=65535)
    alterar_vencimento_faturamento = models.BooleanField(blank=True, null=True)
    utiliza_configuracoes_carencia = models.BooleanField(blank=True, null=True)
    parcela_com_dia_fixo = models.BooleanField(blank=True, null=True)
    prazo_maximo_dias = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'vencimentos'


class VendasContingencia(models.Model):
    codigo = models.AutoField(primary_key=True)
    ordem = models.ForeignKey(CabecalhoOrdemServico, models.DO_NOTHING, db_column='ordem')
    caixa = models.ForeignKey(AbreFechaCaixa, models.DO_NOTHING, db_column='caixa')

    class Meta:
        managed = False
        db_table = 'vendas_contingencia'


class VendasContingenciaPagamento(models.Model):
    codigo = models.AutoField(primary_key=True)
    chave_venda = models.ForeignKey(VendasContingencia, models.DO_NOTHING, db_column='chave_venda')
    especie = models.ForeignKey(Especies, models.DO_NOTHING, db_column='especie')
    valor = models.DecimalField(max_digits=65535, decimal_places=65535)

    class Meta:
        managed = False
        db_table = 'vendas_contingencia_pagamento'


class VendasProspectador(models.Model):
    item = models.OneToOneField(ItensOrdemServico, models.DO_NOTHING, db_column='item', primary_key=True)
    comissao = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'vendas_prospectador'


class VendasRelacaoYandeh(models.Model):
    codigo = models.AutoField(primary_key=True)
    venda = models.ForeignKey(CabecalhoOrdemServico, models.DO_NOTHING, db_column='venda')
    id_yandeh = models.CharField(max_length=20, blank=True, null=True)
    cancelada = models.CharField(max_length=1, blank=True, null=True)
    operador = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='operador', blank=True, null=True)
    data = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'vendas_relacao_yandeh'


class VendedorMetas(models.Model):
    codigo = models.AutoField(primary_key=True)
    codigo_meta = models.ForeignKey(CabecalhoMetas, models.DO_NOTHING, db_column='codigo_meta')
    codigo_vendedor = models.ForeignKey('Vendedores', models.DO_NOTHING, db_column='codigo_vendedor')
    valor = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'vendedor_metas'


class Vendedores(models.Model):
    codigo = models.IntegerField(primary_key=True)
    regiao = models.ForeignKey(Regioes, models.DO_NOTHING, db_column='regiao', blank=True, null=True)
    nome = models.CharField(max_length=50, blank=True, null=True)
    endereco = models.CharField(max_length=50, blank=True, null=True)
    cpf = models.CharField(max_length=14, blank=True, null=True)
    identidade = models.CharField(max_length=20, blank=True, null=True)
    comissao = models.DecimalField(max_digits=5, decimal_places=2, blank=True, null=True)
    cep = models.ForeignKey(Cidades, models.DO_NOTHING, db_column='cep', blank=True, null=True)
    telefone = models.CharField(max_length=14, blank=True, null=True)
    celular = models.CharField(max_length=14, blank=True, null=True)
    email = models.CharField(max_length=50, blank=True, null=True)
    url = models.CharField(max_length=50, blank=True, null=True)
    nivel = models.ForeignKey(NiveisDeAcesso, models.DO_NOTHING, db_column='nivel', blank=True, null=True)
    senha = models.CharField(max_length=15, blank=True, null=True)
    nascimento = models.DateField(blank=True, null=True)
    admissao = models.DateField(blank=True, null=True)
    cargo = models.ForeignKey(Cargos, models.DO_NOTHING, db_column='cargo', blank=True, null=True)
    escala = models.IntegerField(blank=True, null=True)
    bairro = models.CharField(max_length=30, blank=True, null=True)
    comissao_prazo = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    tipo_comissao = models.IntegerField(blank=True, null=True)
    status = models.BooleanField()
    data_palm = models.DateTimeField(blank=True, null=True)
    usa_palm = models.BooleanField(blank=True, null=True)
    usa_saldo_remessa = models.BooleanField()
    tabela_preco = models.ForeignKey(CabecaPreco, models.DO_NOTHING, db_column='tabela_preco', blank=True, null=True)
    crea = models.CharField(max_length=20, blank=True, null=True)
    caixa = models.ForeignKey(CaixaConfiguracao, models.DO_NOTHING, db_column='caixa', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'vendedores'


class VendedoresComissao(models.Model):
    codigo = models.AutoField(primary_key=True)
    vendedor = models.ForeignKey(Vendedores, models.DO_NOTHING, db_column='vendedor', blank=True, null=True)
    comissao = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    desconto_inicial = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    desconto_final = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    novacomissao = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'vendedores_comissao'


class VendedoresConfiguracaoEnvioEmail(models.Model):
    codigo = models.AutoField()
    vendedor = models.ForeignKey(Vendedores, models.DO_NOTHING, db_column='vendedor')
    servidor = models.CharField(max_length=50, blank=True, null=True)
    porta = models.IntegerField(blank=True, null=True)
    usuario = models.CharField(max_length=30, blank=True, null=True)
    senha = models.CharField(max_length=15, blank=True, null=True)
    requer_autenticacao = models.BooleanField(blank=True, null=True)
    utiliza_ssl = models.BooleanField(blank=True, null=True)
    enviar_pdf = models.BooleanField(blank=True, null=True)
    seguranca = models.IntegerField(blank=True, null=True)
    dataoperacao = models.DateTimeField()
    operador = models.ForeignKey(Vendedores, models.DO_NOTHING, db_column='operador')

    class Meta:
        managed = False
        db_table = 'vendedores_configuracao_envio_email'


class VendedoresDescontos(models.Model):
    codigo = models.AutoField(primary_key=True)
    vendedor = models.ForeignKey(Vendedores, models.DO_NOTHING, db_column='vendedor', blank=True, null=True)
    produto = models.ForeignKey(Produtos, models.DO_NOTHING, db_column='produto', blank=True, null=True)
    desconto = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'vendedores_descontos'


class VigenciaContrato(models.Model):
    codigo = models.AutoField(primary_key=True)
    cd_ordem = models.ForeignKey(CabecalhoOrdemServico, models.DO_NOTHING, db_column='cd_ordem', blank=True, null=True)
    data_inicial = models.DateField(blank=True, null=True)
    data_final = models.DateField(blank=True, null=True)
    controle_interno = models.IntegerField(blank=True, null=True)
    data_finalizado = models.DateField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'vigencia_contrato'


class ZeramentoSaldoItensCompra(models.Model):
    cd_item = models.OneToOneField(ItensCompra, models.DO_NOTHING, db_column='cd_item', primary_key=True)
    quantidade = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    usuario = models.ForeignKey(Vendedores, models.DO_NOTHING, db_column='usuario')
    data = models.DateField()
    hora = models.TimeField()

    class Meta:
        managed = False
        db_table = 'zeramento_saldo_itens_compra'
